from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn import manifold
import pickle
import os
from typing import List


def allocate(n: int, m: int, M_sets: list, D_len: list):
    u = len(M_sets)

    assert np.max(np.array(D_len)) <= n
    assert len(M_sets) == len(D_len)

    for M_set in M_sets:
        if len(M_set) > m or len(M_set) == 0:
            assert False

    A_matrix = np.zeros((n, m))
    A_matrix = A_matrix - 1

    # print(M_set, D_len)
    for k in range(u):
        i = 0
        allocated = 0
        while i < n and allocated < D_len[k]:
            row = [A_matrix[i][j] for j in M_sets[k]]
            if np.max(row) == -1:
                allocated += 1
                for j in M_sets[k]:
                    A_matrix[i][j] = k
            i += 1

    return A_matrix.T


def distribute_data_idx(client_nums_for_Dk: list, D_ks: list):
    D_kps = []
    for k, D_k in enumerate(D_ks, 0):
        D_kp_id = []
        D_k_X, D_k_Y = D_k
        data_len = len(D_k_Y)
        data_num_p = data_len // client_nums_for_Dk[k]
        for p in range(client_nums_for_Dk[k]):
            D_kp_X_id = [
                D_k_X[i][p * data_num_p : (p + 1) * data_num_p]
                for i in range(len(D_k_X))
            ]

            D_kp_Y_id = D_k_Y[p * data_num_p : (p + 1) * data_num_p]
            D_kp_id.append((D_kp_X_id, D_kp_Y_id))
            # print(D_kp_X_id[0][:100])
            # print(D_kp_Y_id[:100])
            # assert False
        D_kps.append(D_kp_id)
    return D_kps


def allocate_data_idx(
    client_nums_for_Dk: list,
    data_len: int,
    M_sets: list,
    M_num: int,
    Dk_len: list,
    data_idx_list: List[list],
):
    A_matrix = allocate(data_len, M_num, M_sets, Dk_len)
    D_ks = []
    for k, subset in enumerate(M_sets, 0):
        D_k = []
        data_indexes = None
        for modality_index in subset:
            tmp = np.argwhere(A_matrix[modality_index] == k).flatten()
            if data_indexes is not None:
                assert all([a == b for a, b in zip(data_indexes, tmp)])
            data_indexes = tmp
            data_X = [data_idx_list[modality_index][idx] for idx in data_indexes]
            D_k.append(data_X)
        data_Y = [
            os.path.basename(os.path.dirname(data_idx_list[0][idx]))
            for idx in data_indexes
        ]

        D_ks.append([D_k, data_Y])
    D_kps = distribute_data_idx(client_nums_for_Dk, D_ks)
    return D_kps


def distribute_dataset(client_nums_for_Dk: list, D_ks: list):
    D_kps = []
    for k, D_k in enumerate(D_ks, 0):
        D_k_X, D_k_Y = D_k
        D_k = []
        data_len = len(D_k_X[0])
        data_num_p = data_len // client_nums_for_Dk[k]
        for p in range(client_nums_for_Dk[k]):
            D_kp_X_mm = []
            for m in range(len(D_k_X)):
                D_kp_X_mm.append(D_k_X[m][p * data_num_p : (p + 1) * data_num_p])
            D_kp_Y = D_k_Y[p * data_num_p : (p + 1) * data_num_p]
            D_k.append((D_kp_X_mm, D_kp_Y))
        D_kps.append(D_k)
    return D_kps


def get_splitted_datasets(
    data_X_: List[np.ndarray],
    data_Y_: List[np.ndarray],
    data_len: int,
    M_sets: list,
    M_num: int,
    D_len: list,
    client_nums_for_Dk,
    shuffle=False,
):
    data_X = None
    data_Y = None
    if shuffle:
        data_num = len(data_X_[0])
        shuffle_idx = np.random.permutation(data_num)
        data_X = []
        for i in range(len(data_X_)):
            data_X.append(data_X_[i][shuffle_idx, :, :])
        data_Y = data_Y_[shuffle_idx]
    else:
        data_X = data_X_
        data_Y = data_Y_

    N = data_len
    M = M_num
    A_matrix = allocate(N, M, M_sets, D_len)
    # print(A_matrix.T)
    D_ks = []
    for k, subset in enumerate(M_sets, 0):
        D_k = []
        data_indexes = None
        for modality_index in subset:
            tmp = np.argwhere(A_matrix[modality_index] == k).flatten()
            if data_indexes is not None:
                assert all([a == b for a, b in zip(data_indexes, tmp)])
            data_indexes = tmp
            D_k.append(data_X[modality_index][data_indexes])
            # print(modality_index, data_X[modality_index][data_indexes].shape)
        D_ks.append((D_k, data_Y[data_indexes]))
    D_kps = distribute_dataset(client_nums_for_Dk, D_ks)

    return D_kps


def get_datadir(dataset_name, split_name):
    datasetDir = "data/" + dataset_name
    picklesDir = "data/" + dataset_name + "/" + split_name

    return datasetDir, picklesDir


def init_data_dir(datasetDir, picklesDir):
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir(datasetDir):
        os.mkdir(datasetDir)
    if os.path.isdir(picklesDir):
        os.system(f"rm -rf {picklesDir}")
    os.mkdir(f"{picklesDir}")


def data_generating_store(
    picklesDir: str,
    file_name: str,
    data: dict,
    clientsInfo: dict,
):
    with open(picklesDir + "/" + file_name + ".pkl", "wb") as f:
        pickle.dump(data, f)
    with open(picklesDir + "/" + "seperation.pkl", "wb") as f:
        # 在这个地方划分测试集和训练集
        pickle.dump(
            clientsInfo,
            f,
        )


class basic_dataset(Dataset):
    def __init__(self, M_set, data_list, targets, segment_list=None):
        self.M_set = M_set
        self.data0 = None
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.segment_list = segment_list
        if len(M_set) == 0:
            assert False
        elif len(M_set) == 1:
            self.data0 = data_list[0]
        elif len(M_set) == 2:
            self.data0 = data_list[0]
            self.data1 = data_list[1]
        elif len(M_set) == 3:
            self.data0 = data_list[0]
            self.data1 = data_list[1]
            self.data2 = data_list[2]
        elif len(M_set) == 4:
            self.data0 = data_list[0]
            self.data1 = data_list[1]
            self.data2 = data_list[2]
            self.data3 = data_list[3]
        else:
            assert False
        self.targets = targets

    def __len__(self):
        return self.data0.shape[0]

    def __getitem__(self, index):
        if len(self.M_set) == 0:
            assert False
        elif len(self.M_set) == 1:
            return self.data0[index], self.targets[index]
        elif len(self.M_set) == 2:
            return self.data0[index], self.data1[index], self.targets[index]
        elif len(self.M_set) == 3:
            return (
                self.data0[index],
                self.data1[index],
                self.data2[index],
                self.targets[index],
            )

        elif len(self.M_set) == 4:
            return (
                self.data0[index],
                self.data1[index],
                self.data2[index],
                self.data3[index],
                self.targets[index],
            )
        else:
            assert False


class sequence_dataset(Dataset):
    def __init__(self, basic: basic_dataset, seq_len: int):
        self.basic = basic
        self.M_set = basic.M_set
        self.seq_len = seq_len
        self.targets = None
        # print("here", self.basic.M_set)
        if len(self.basic.M_set) == 0:
            assert False
        elif len(self.basic.M_set) == 1:
            self.data0 = self.make_sequences(self.basic.data0, self.seq_len)
        elif len(self.basic.M_set) == 2:
            self.data0 = self.make_sequences(self.basic.data0, self.seq_len)
            self.data1 = self.make_sequences(self.basic.data1, self.seq_len)
        elif len(self.basic.M_set) == 3:
            self.data0 = self.make_sequences(self.basic.data0, self.seq_len)
            self.data1 = self.make_sequences(self.basic.data1, self.seq_len)
            self.data2 = self.make_sequences(self.basic.data2, self.seq_len)
        elif len(self.basic.M_set) == 4:
            self.data0 = self.make_sequences(self.basic.data0, self.seq_len)
            self.data1 = self.make_sequences(self.basic.data1, self.seq_len)
            self.data2 = self.make_sequences(self.basic.data2, self.seq_len)
            self.data3 = self.make_sequences(self.basic.data3, self.seq_len)
        else:
            assert False

    def make_sequences(self, data_X, seq_len):
        new_data = []
        new_targets = []
        sample_id_in_seg_start = 0
        for segment_id, segment_len in self.basic.segment_list:
            seq_num_in_seg = segment_len // seq_len
            if seq_num_in_seg == 0:
                continue
            for seq_id_in_seg in range(seq_num_in_seg):

                data_X_ = data_X[
                    sample_id_in_seg_start
                    + seq_id_in_seg * seq_len : sample_id_in_seg_start
                    + (seq_id_in_seg + 1) * seq_len
                ]
                # print(data_X_.shape)
                new_data.append(data_X_)
                if self.targets is None:
                    new_targets.append(
                        self.basic.targets[
                            sample_id_in_seg_start
                            + seq_id_in_seg * seq_len : sample_id_in_seg_start
                            + (seq_id_in_seg + 1) * seq_len
                        ]
                    )
            sample_id_in_seg_start += segment_len
        seq_data = torch.tensor(np.array(new_data), dtype=torch.float32)
        # print(seq_data.shape)
        if self.targets is None:
            self.targets = torch.tensor(np.array(new_targets), dtype=torch.int64)
        return seq_data

    def __len__(self):
        return self.data0.shape[0]

    def __getitem__(self, index):
        if len(self.M_set) == 0:
            assert False
        elif len(self.M_set) == 1:
            return self.data0[index], self.targets[index]
        elif len(self.M_set) == 2:
            return self.data0[index], self.data1[index], self.targets[index]
        elif len(self.M_set) == 3:
            return (
                self.data0[index],
                self.data1[index],
                self.data2[index],
                self.targets[index],
            )

        elif len(self.M_set) == 4:
            return (
                self.data0[index],
                self.data1[index],
                self.data2[index],
                self.data3[index],
                self.targets[index],
            )
        else:
            assert False


def load_dataloaders(dataset_name, split_name, file_name):
    _, pickles_dir = get_datadir(dataset_name, split_name)
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(pickles_dir + "/" + file_name + ".pkl", "rb") as f:
        dataloaders = pickle.load(f)
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]
