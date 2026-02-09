from scipy.io import savemat, loadmat
from scipy.stats import zscore
import numpy as np
from torch.utils.data import DataLoader
from data_generating.utils import (
    basic_dataset,
    sequence_dataset,
    data_generating_store,
    get_datadir,
    init_data_dir,
)
from control.Enums import ParametersForDataSplitType, generate_client_id
import torch


def load_URFALL_data():
    modalities = ["acce", "rgb", "depth"]
    mat_data = loadmat("data/URFALL/urfall.mat")
    fall_test = [1, 6, 10, 20, 28]  # np.random.choice(range(1, 31), 3, replace=False)
    adl_test = [
        2,
        5,
        11,
        15,
        17,
        25,
        29,
        38,
    ]  # np.random.choice(range(1, 41), 4, replace=False)
    test_vids = [adl_test, fall_test]
    data_train = {0: [], 1: [], 2: [], "y": [], "segment_indices": []}
    data_test = {0: [], 1: [], 2: [], "y": [], "segment_indices": []}

    # fall videos
    vid_id_list = [(1, i) for i in range(1, 31)] + [(0, i) for i in range(1, 41)]

    segment_train_id = 0
    segment_test_id = 0
    for is_fall, v_number in vid_id_list:
        a_y = mat_data["y"]
        sub_a_y = a_y[(a_y[:, 0] == is_fall) & (a_y[:, 1] == v_number), :]
        sub_a_y = sub_a_y[:, 3]
        for m_idx, modality_textual_name in enumerate(modalities, 0):
            a_X = mat_data[modality_textual_name]
            sub_a_X = a_X[(a_X[:, 0] == is_fall) & (a_X[:, 1] == v_number), :]
            if modality_textual_name == "acce" or modality_textual_name == "depth":
                sub_a_X[:, 3:] = zscore(sub_a_X[:, 3:])
            sub_a_X = sub_a_X[:, 3:]
            if v_number in test_vids[is_fall]:
                data_test[m_idx].append(sub_a_X)
            else:
                data_train[m_idx].append(sub_a_X)
        if v_number in test_vids[is_fall]:
            data_test["y"].append(sub_a_y)
            data_test["segment_indices"].append((segment_test_id, len(sub_a_y)))
            segment_test_id += 1
        else:
            data_train["y"].append(sub_a_y)
            data_train["segment_indices"].append((segment_train_id, len(sub_a_y)))
            segment_train_id += 1

    for m_idx in range(len(modalities)):
        data_train[m_idx] = np.concatenate(data_train[m_idx])
        data_test[m_idx] = np.concatenate(data_test[m_idx])
    data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
    data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
    return (data_train, data_test)


def make_sequences(data_X, segment_list, seq_len, targets=None):
    new_data = []
    new_targets = []
    sample_id_in_seg_start = 0
    for segment_id, segment_len in segment_list:
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
            if targets is not None:
                new_targets.append(
                    targets[
                        sample_id_in_seg_start
                        + seq_id_in_seg * seq_len : sample_id_in_seg_start
                        + (seq_id_in_seg + 1) * seq_len
                    ]
                )
        sample_id_in_seg_start += segment_len
    seq_data = np.array(new_data)
    # print(seq_data.shape)
    new_targets = np.array(new_targets)
    return (seq_data, new_targets)


def URFALL_data_generating():
    train, test = load_URFALL_data()
    data_X_train = [
        make_sequences(train[m], train["segment_indices"], 30)[0] for m in range(3)
    ]
    data_Y_train = make_sequences(train[0], train["segment_indices"], 30, train["y"])[1]

    data_X_test = [
        make_sequences(test[m], test["segment_indices"], 30)[0] for m in range(3)
    ]
    data_Y_test = make_sequences(test[0], test["segment_indices"], 30, test["y"])[1]
    return data_X_train, data_Y_train, data_X_test, data_Y_test
    # train_loader = DataLoader(
    #     basic_dataset(
    #         split_parameters.M_sets[0],
    #         data_X_train,
    #         data_Y_train,
    #         segment_list=train["segment_indices"],
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    # )
    # test_loader = DataLoader(
    #     basic_dataset(
    #         split_parameters.M_sets[0],
    #         data_X_test,
    #         data_Y_test,
    #         segment_list=test["segment_indices"],
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    # )

    # data_generating_store(
    #     picklesDir,
    #     "0_0_",
    #     {"train": train_loader, "test": test_loader, "val": test_loader},
    #     {},
    # )

    # print(b.data0.shape)
    # print(b.targets.shape)

    # for i in range(len(train[0])):
    #     print(train[0][i].shape)
    #     print(train[1][i].shape)
    #     print(train[2][i].shape)
    #     print(train["y"][i].shape)
    #     print(train["segment_indices"][i])
    #     print("....")

    # print(train[0].shape)
    # print(train[1].shape)
    # print(train[2].shape)
    # print(train["y"].shape)
    # print(sum([a[1] for a in train["segment_indices"]]))
    # print(train["segment_indices"])
    # print("....")
