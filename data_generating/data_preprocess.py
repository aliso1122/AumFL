from data_generating.utils import basic_dataset
from torch.utils.data import DataLoader
from control.Enums import ParametersForDataSplitType, SpeiclaClientType, DatasetName
import os
import torch
from data_generating.HAR import HAR_data_generating
from data_generating.UCF101 import UCF101_create_dataloaders
from data_generating.URFALL import URFALL_data_generating
from data_generating.CMUMOSEI import CMUMOSEI_data_generating
import numpy as np
from data_generating.utils import (
    load_dataloaders,
    get_datadir,
    init_data_dir,
    get_splitted_datasets,
    data_generating_store,
)
import random


def create_(args):
    datasetDir, picklesDir = get_datadir(args.dataset_name, args.data_split_type)
    init_data_dir(datasetDir, picklesDir)
    train_X, train_Y, test_X, test_Y = CMUMOSEI_data_generating()
    dataloader_train = DataLoader(
        basic_dataset(
            (0, 1, 2),
            [torch.tensor(x, dtype=torch.float32) for x in train_X],
            torch.tensor(train_Y, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    dataloader_test = DataLoader(
        basic_dataset(
            (0, 1, 2),
            [torch.tensor(x, dtype=torch.float32) for x in test_X],
            torch.tensor(test_Y, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    data_generating_store(
        picklesDir,
        f"{0}_{0}_",
        {
            "train": dataloader_train,
            "test": dataloader_test,
            "val": dataloader_test,
        },
        {},
    )


# HAR dataloader batch: Tensor(batch_size, seq_len=128, input_size=3)
def create(args):
    datasetDir, picklesDir = get_datadir(args.dataset_name, args.data_split_type)
    init_data_dir(datasetDir, picklesDir)
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None
    target_dtype = torch.int32
    if args.dataset_name == DatasetName.HAR.value:
        train_X, train_Y, test_X, test_Y = HAR_data_generating()
    elif args.dataset_name == DatasetName.URFALL.value:
        train_X, train_Y, test_X, test_Y = URFALL_data_generating()
    elif args.dataset_name == DatasetName.CMUMOSEI.value:
        target_dtype = torch.float32
        train_X, train_Y, test_X, test_Y = CMUMOSEI_data_generating()

    N_train = len(train_X[0])
    print("N_train:", N_train)
    for m in range(1, len(train_X)):
        assert N_train == len(train_X[m])
    assert N_train == len(train_Y)
    N_test = len(test_X[0])
    for m in range(1, len(test_X)):
        assert N_test == len(test_X[m])
    assert N_test == len(test_Y)

    split_parameters = ParametersForDataSplitType(args.data_split_type, N_train, N_test)
    M_sets = split_parameters.M_sets
    D_len_train = split_parameters.D_len_train
    D_len_test = split_parameters.D_len_test
    client_nums_for_Dk = split_parameters.client_nums_for_Dk
    D_kps_train = get_splitted_datasets(
        train_X,
        train_Y,
        N_train,
        M_sets,
        3,
        D_len_train,
        client_nums_for_Dk,
    )
    D_kps_test = get_splitted_datasets(
        test_X,
        test_Y,
        N_test,
        M_sets,
        3,
        D_len_test,
        client_nums_for_Dk,
    )
    for k, D_k_train in enumerate(D_kps_train, 0):
        D_k_test = D_kps_test[k]
        for p, D_p_train in enumerate(D_k_train, 0):
            D_p_test = D_k_test[p]
            data_train, targets_train = D_p_train
            data_test, targets_test = D_p_test
            dataloader_train = DataLoader(
                basic_dataset(
                    M_sets[k],
                    [torch.tensor(x, dtype=torch.float32) for x in data_train],
                    torch.tensor(targets_train, dtype=target_dtype),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            dataloader_test = DataLoader(
                basic_dataset(
                    M_sets[k],
                    [torch.tensor(x, dtype=torch.float32) for x in data_test],
                    torch.tensor(targets_test, dtype=target_dtype),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            data_generating_store(
                picklesDir,
                f"{k}_{p}_",
                {
                    "train": dataloader_train,
                    "test": dataloader_test,
                    "val": dataloader_test,
                },
                {},
            )
            print(
                f"created dataloaders for client{k}-{p} and the data shapes are",
                ",".join([str(x.shape) for x in data_train]),
                "and",
                ",".join([str(x.shape) for x in data_test]),
                "for traning and testing",
            )

    data_global_test, targets_global_test = get_splitted_datasets(
        test_X,
        test_Y,
        N_test,
        (split_parameters.global_m_set,),
        3,
        [N_test],
        [1],
    )[0][0]
    dataloader_global_test = DataLoader(
        basic_dataset(
            tuple(split_parameters.global_m_set),
            [torch.tensor(x, dtype=torch.float32) for x in data_global_test],
            torch.tensor(targets_global_test, dtype=target_dtype),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    data_global_train, targets_global_train = get_splitted_datasets(
        train_X,
        train_Y,
        N_train,
        (split_parameters.global_m_set,),
        3,
        [N_train],
        [1],
    )[0][0]
    dataloader_global_train = DataLoader(
        basic_dataset(
            tuple(split_parameters.global_m_set),
            [torch.tensor(x, dtype=torch.float32) for x in data_global_train],
            torch.tensor(targets_global_train, dtype=target_dtype),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    data_generating_store(
        picklesDir,
        SpeiclaClientType.GLOBAL_CLIENT.value,
        {"test": dataloader_global_test, "train": dataloader_global_train, "val": None},
        {},
    )
    print(
        "created test-dataloader for global server and the data shape is",
        ",".join([str(x.shape) for x in data_global_train]),
    )


def create_dataloaders(args):
    if args.dataset_name == DatasetName.HAR.value:
        create(args)
    elif args.dataset_name == DatasetName.UCF101.value:
        UCF101_create_dataloaders(args)
    elif args.dataset_name == DatasetName.URFALL.value:
        create(args)
    elif args.dataset_name == DatasetName.CMUMOSEI.value:
        create(args)
    else:
        assert False


def get_dataloaders(dataset_name, spilt_name, file_name):
    return load_dataloaders(dataset_name, spilt_name, file_name)
