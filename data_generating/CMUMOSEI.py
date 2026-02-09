import h5py

import numpy as np
import torchtext as text
from collections import defaultdict
from data_generating.utils import basic_dataset
import pickle
from torch.utils.data import DataLoader
import torch


def lpad(this_array, seq_len):
    """Left pad array with seq_len 0s.

    Args:
        this_array (np.array): Array to pad
        seq_len (int): Number of 0s to pad.

    Returns:
        np.array: Padded array
    """

    temp_array = np.concatenate(
        [np.zeros([seq_len] + list(this_array.shape[1:])), this_array], axis=0
    )[-seq_len:, ...]
    return temp_array


def detect_entry_fold(entry, standard_train_fold):
    """Detect entry fold.

    Args:
        entry (str): Entry string
        folds (int): Number of folds

    Returns:
        int: Entry fold index
    """
    entry_id = entry.split("[")[0]
    if entry_id in standard_train_fold:
        return 0
    else:
        return 1
    # for i in range(len(folds)):
    #     if entry_id in folds[i]:
    #         return i

    return None


def get_rawtext(path, data_kind, vids):
    """Get raw text modality.

    Args:
        path (str): Path to h5 file
        data_kind (str): String for data format. Should be 'hdf5'.
        vids (list): List of video ids.

    Returns:
        tuple(list,list): Tuple of text_data and video_data in lists.
    """
    if data_kind == "hdf5":
        f = h5py.File(path, "r")
        text_data = []
        new_vids = []
        count = 0
        for vid in vids:
            text = []
            # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
            # vid_id = '{}[{}]'.format(id, seg)
            vid_id = vid
            # TODO: fix 31 missing entries
            try:
                for word in f["words"][vid_id]["features"]:
                    if word[0] != b"sp":
                        text.append(word[0].decode("utf-8"))
                text_data.append(" ".join(text))
                new_vids.append(vid_id)
            except:
                print("missing", vid, vid_id)
        return text_data, new_vids
    else:
        print("Wrong data kind!")


def get_word2id(text_data, vids):
    """From text_data, vids get word2id lsit

    Args:
        text_data (list): List of text data
        vids (list): List of video data

    Returns:
        list: List of word2id data
    """

    word2id = defaultdict(lambda: len(word2id))
    # print(word2id)
    UNK = word2id["unk"]
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()

        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def _return_unk():
        return UNK

    word2id.default_factory = _return_unk
    return data_processed, word2id


def get_word_embeddings(word2id, save=False):
    """Given a word2id, get the associated glove embeddings ( 300 dimensional ).

    Args:
        word2id (list): list of word, index pairs
        save (bool, optional): Whether to save data to the folder (unused). Defaults to False.

    Returns:
        list[np.array]: List of embedded words
    """
    vec = text.vocab.GloVe(name="840B", dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)

    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def glove_embeddings(text_data, vids, paddings=50):
    """Get glove embeddings of text, video pairs.

    Args:
        text_data (list): list of text data.
        vids (list): list of video data
        paddings (int, optional): Amount to left-pad data if it's less than some size. Defaults to 50.

    Returns:
        np.array: Array of embedded data
    """
    data_prod, w2id = get_word2id(text_data, vids)
    word_embeddings_looks_up = get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        # TODO: fix some segs have more than 50 words
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(
                    np.zeros(
                        300,
                    )
                )
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:

        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


def get_audio_visual_text(
    csds, seq_len, text_data, vids, affect_data, labels, folds, fold_names
):
    """Get audio visual from text."""
    data = [{} for _ in range(len(fold_names))]
    output = [{} for _ in range(len(fold_names))]

    for i in range(len(folds)):
        for csd in csds:
            data[i][csd] = []
        data[i]["words"] = []
        data[i]["id"] = []

    for i, key in enumerate(vids):
        which_fold = detect_entry_fold(key, folds[0])
        # print(which_fold)
        # print("which_fold:", which_fold)
        if which_fold == None:
            # print("Key {} doesn't belong to any fold ... ".format(key))
            continue
        for csd in csds:
            this_array = affect_data[csd][key]["features"]
            if csd in labels:
                data[which_fold][csd].append(this_array)
            else:
                data[which_fold][csd].append(lpad(this_array, seq_len=seq_len))

        data[which_fold]["words"].append(text_data[i])
        data[which_fold]["id"].append(key)

    for i in range(len(folds)):
        for csd in csds:
            output[i][csd] = np.array(data[i][csd])
        output[i]["words"] = np.stack(data[i]["words"])
        output[i]["id"] = data[i]["id"]

    for i in range(len(fold_names)):
        for csd in csds:
            print(
                "Shape of the %s computational sequence for %s fold is %s"
                % (csd, fold_names[i], output[i][csd].shape)
            )
        print(
            "Shape of the %s computational sequence for %s fold is %s"
            % ("words", fold_names[i], output[i]["words"].shape)
        )
    return output


def CMUMOSEI_data_generating():
    with open("data/CMUMOSEI/raw/train.txt") as f:
        standard_train_fold = list(
            map(lambda x: x.strip("\n").strip(",")[1:-1], f.readlines())
        )
    with open("data/CMUMOSEI/raw/test.txt") as f:
        standard_test_fold = list(
            map(lambda x: x.strip("\n").strip(",")[1:-1], f.readlines())
        )
    # print(standard_train_fold)

    # assert False
    folds = [standard_train_fold, standard_test_fold]
    fold_names = ["train", "test"]
    affect_data = h5py.File("data/CMUMOSEI/raw/mosei.hdf5", "r")
    # print(affect_data.keys())
    AUDIO = "COVAREP"
    VIDEO = "OpenFace_2"
    WORD = "words"
    labels = ["All Labels"]

    csds = [AUDIO, VIDEO, labels[0]]
    channels = [AUDIO, VIDEO, WORD]
    seq_len = 50

    keys = list(affect_data[WORD].keys())

    raw_text, vids = get_rawtext("data/CMUMOSEI/raw/mosei.hdf5", "hdf5", keys)
    # max_v = -10000
    # min_v = 10000
    # counter = {}
    # for i in range(len(vids)):
    #     label_vec = np.array(affect_data["All Labels"][vids[i]]["features"])[0]
    #     sub_vec = label_vec[:1]
    #     for value in sub_vec:
    #         if value in counter:
    #             counter[value] += 1
    #         else:
    #             counter[value] = 1
    #     if max_v < np.max(sub_vec):
    #         max_v = np.max(sub_vec)
    #         print(label_vec)
    #     if min_v > np.min(sub_vec):
    #         min_v = np.min(sub_vec)
    #         print(label_vec)
    # keys = list(counter.keys())
    # keys.sort()
    # for k in keys:
    #     print(k, counter[k])

    # print(vids[i], np.array(affect_data["All Labels"][vids[i]]["features"]))
    # print(min_v, max_v)

    text_glove = glove_embeddings(raw_text, vids)
    # print("text", text_glove.shape)
    audio_video_text = get_audio_visual_text(
        csds,
        seq_len=seq_len,
        text_data=text_glove,
        vids=vids,
        affect_data=affect_data,
        labels=labels,
        folds=folds,
        fold_names=fold_names,
    )

    train_X = [audio_video_text[0][c] for c in channels]
    train_Y_ = audio_video_text[0][labels[0]]
    train_Y = np.array([[a[0][0]] for a in train_Y_])

    test_X = [audio_video_text[1][c] for c in channels]
    test_Y_ = audio_video_text[1][labels[0]]
    test_Y = np.array([[a[0][0]] for a in test_Y_])
    return train_X, train_Y, test_X, test_Y
    # train_loader = DataLoader(
    #     basic_dataset(
    #         (1, 2),
    #         [torch.tensor(train_data[c], dtype=torch.float) for c in channels[1:]],
    #         train_data[labels[0]],
    #     ),
    #     batch_size=16,
    #     shuffle=True,
    # )

    # with open("train.pkl", "wb") as f:
    #     pickle.dump(
    #         train_loader,
    #         f,
    #     )
