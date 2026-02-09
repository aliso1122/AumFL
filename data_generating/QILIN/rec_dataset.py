from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import random
from data_generating.QILIN.tools import load_Qilin_data
import pandas as pd
class DataProcessor:
    def __init__(self, **kwargs):
        """
        Data processor class for DCN search model, used for loading and processing training data.

        Args:
            dataset_name_or_path (str): Dataset path
            batch_size (int): Batch size
            negative_samples (int): Number of negative samples per query
        """
        self.batch_size = kwargs.get('batch_size', 8)
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.max_length = kwargs.get('max_length', 512)
        self.use_title = kwargs.get('use_title',1)
        self.use_content = kwargs.get('use_content',1)
        # tokenizer_name = kwargs.get('tokenizer_name_or_path')
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # self.tokenizer.truncation_side = 'right'
        # self.tokenizer.padding_side = 'right'
        # self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret
    
    def load_data(self):
        """
        Load dataset from disk
        """
        self.corpus = load_Qilin_data("notes")
        self.dataset = load_Qilin_data("recommendation_train")
        self.user_features = load_Qilin_data("user_feat")

    def get_note_dense_features(self, note_idx):
        """
        Get dense features of the note
        """
        note = self.corpus[note_idx]
        note_dense_feature_names = ['video_duration', 'video_height', 'video_width', 'image_num', 
                                    'content_length', 'commercial_flag', 'imp_num', 'imp_rec_num', 
                                    'imp_search_num', 'click_num', 'click_rec_num', 'click_search_num', 
                                    'like_num', 'collect_num', 'comment_num', 'share_num', 'screenshot_num',
                                    'hide_num', 'rec_like_num', 'rec_collect_num', 'rec_comment_num', 
                                    'rec_share_num', 'rec_follow_num', 'search_like_num', 
                                    'search_collect_num', 'search_comment_num', 'search_share_num', 
                                    'search_follow_num', 'accum_like_num', 'accum_collect_num', 
                                    'accum_comment_num', 'view_time', 'rec_view_time', 'search_view_time', 
                                    'valid_view_times', 'full_view_times']
        features = [note[feature_name] for feature_name in note_dense_feature_names]
        features = [0.0 if pd.isna(x) else x for x in features]
        return torch.tensor(features, dtype=torch.float32)

    def get_note_sparse_features(self, note_idx):
        """
        Get sparse features of the note
        """
        note = self.corpus[note_idx]
        return {
            'note_type': torch.tensor(note['note_type'], dtype=torch.long),
            'taxonomy1_id': torch.tensor(hash(note['taxonomy1_id']) % 43, dtype=torch.long),
            'taxonomy2_id': torch.tensor(hash(note['taxonomy2_id']) % 311, dtype=torch.long),
            'taxonomy3_id': torch.tensor(hash(note['taxonomy3_id']) % 548, dtype=torch.long),
            'note_idx': torch.tensor(note_idx, dtype=torch.long)
        }

    def get_user_dense_features(self, user_idx):
        """
        Get user dense feature
        """
        user = self.user_features[user_idx]
        dense_features = [user[f'dense_feat{i}'] for i in range(1, 41)]
        dense_features.extend([user['fans_num'], user['follows_num']])
        dense_features = [0.0 if pd.isna(x) else x for x in dense_features]
        return torch.tensor(dense_features, dtype=torch.float32)

    def get_user_sparse_features(self, user_idx):
        """
        Get user sparse feature
        """
        user = self.user_features[user_idx]
        gender_map = {'male': 0, 'female': 1, 'unknown': 2}
        platform_map = {'iOS': 0, 'Android': 1, 'Harmony': 2, 'unknown': 3}
        age_map = {
            '1-12': 0, '13-15': 1, '16-18': 2, '19-22': 3, '23-25': 4,
            '26-30': 5, '31-35': 6, '36-40': 7, '40+':8, 'unknown': 9
        }
        return {
            'gender': torch.tensor(gender_map.get(user['gender'], 2), dtype=torch.long),
            'platform': torch.tensor(platform_map.get(user['platform'], 3), dtype=torch.long),
            'age': torch.tensor(age_map.get(user['age'], 9), dtype=torch.long),
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'location': torch.tensor(hash(user['location']) % 1096 if user['location'] else 0, dtype=torch.long)
        }

    def collate_fn(self, batch):
        """
        Batch data processing function
        """
        query_features = {'question_embedding': [], 'query_from_type': []}
        user_features = {'dense': [], 'recent_clicked_note_idxs': []}
        note_features = {'note_embedding': [], 'dense': []}
        labels = []

        for item in batch:
            # process positive and negative samples
            impression_result_details = item[self.negative_pool]
            positives = [r['note_idx'] for r in impression_result_details if r['click'] == 1]
            negatives = [r['note_idx'] for r in impression_result_details if r['click'] == 0]

            # process positive sample
            if positives:
                pos_idx = random.choice(positives)
                query_features['query_from_type'].append(torch.tensor(item.get('query_from_type', 15), dtype=torch.long))
                query_features['question_embedding'].append(item['question_embedding'])

                user_idx = item['user_idx']
                user_dense = self.get_user_dense_features(user_idx)
                user_sparse = self.get_user_sparse_features(user_idx)
                user_features['dense'].append(user_dense)
                for k, v in user_sparse.items():
                    if k not in user_features:
                        user_features[k] = []
                    user_features[k].append(v)
                
                recent_notes = item['recent_clicked_note_idxs'][:10]  # 只取前10个
                if len(recent_notes) < 10:
                    recent_notes = recent_notes + [1983938] * (10 - len(recent_notes))  # 补1983938作为填充
                user_features['recent_clicked_note_idxs'].append(torch.tensor(recent_notes))

                note_dense = self.get_note_dense_features(pos_idx)
                note_sparse = self.get_note_sparse_features(pos_idx)
                note_features['dense'].append(note_dense)
                note_features['note_embedding'].append(self.corpus[pos_idx]['note_embedding'])
                for k, v in note_sparse.items():
                    if k not in note_features:
                        note_features[k] = []
                    note_features[k].append(v)
                labels.append(1)

                if len(negatives) < self.negative_samples:
                    additional = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                    negatives.extend(additional)
                selected_negs = random.sample(negatives, k=self.negative_samples)
                
                for neg_idx in selected_negs:
                    query_features['query_from_type'].append(torch.tensor(item.get('query_from_type', 15), dtype=torch.long))
                    query_features['question_embedding'].append(item['question_embedding'])

                    user_features['dense'].append(user_dense)
                    for k, v in user_sparse.items():
                        user_features[k].append(v)
                    user_features['recent_clicked_note_idxs'].append(torch.tensor(recent_notes))

                    note_dense = self.get_note_dense_features(neg_idx)
                    note_sparse = self.get_note_sparse_features(neg_idx)
                    note_features['dense'].append(note_dense)
                    note_features['note_embedding'].append(self.corpus[neg_idx]['note_embedding'])
                    for k, v in note_sparse.items():
                        note_features[k].append(v)
                    labels.append(0)

        query_features['query_from_type'] = torch.stack([torch.tensor(x) for x in query_features['query_from_type']])
        query_features['question_embedding'] = torch.stack([torch.tensor(x) for x in query_features['question_embedding']])
        user_features['dense'] = torch.stack(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = torch.stack(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = torch.stack(user_features[k])
        note_features['dense'] = torch.stack(note_features['dense'])
        note_features['note_embedding'] = torch.stack([torch.tensor(x) for x in note_features['note_embedding']])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = torch.stack(note_features[k])
        labels = torch.tensor(labels, dtype=torch.float32)
        
        def check_and_fix_nan(tensor):
            if torch.isnan(tensor).any():
                return torch.nan_to_num(tensor, nan=0.0)
            return tensor

        query_features['query_from_type'] = check_and_fix_nan(query_features['query_from_type'])

        user_features['dense'] = check_and_fix_nan(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = check_and_fix_nan(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = check_and_fix_nan(user_features[k])

        note_features['dense'] = check_and_fix_nan(note_features['dense'])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = check_and_fix_nan(note_features[k])

        labels = check_and_fix_nan(labels)
        
        return query_features, user_features, note_features, labels

    def get_dataloader(self):
        """
        Get DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )