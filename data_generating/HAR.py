import numpy as np
from numpy import dstack
from pandas import read_csv
import torch
import numpy as np
from data_generating.utils import basic_dataset
from torch.utils.data import DataLoader
from control.Enums import ParametersForDataSplitType, SpeiclaClientType, DatasetName
import torch

from data_generating.UCF101 import frames_extraction
import numpy as np
from data_generating.utils import (
    data_generating_store,
    get_datadir,
    load_dataloaders,
    init_data_dir,
    get_splitted_datasets,
    allocate_data_idx,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=""):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(np.array(data))
    # stack group so that features are the 3rd dimension, (features x samples x seqlen ) -> (samples x seqlen x features)
    loaded = dstack(loaded)
    print("loaded:", loaded.shape)

    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=""):
    filepath = prefix + group + "/Inertial Signals/"
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += [
        "total_acc_x_" + group + ".txt",
        "total_acc_y_" + group + ".txt",
        "total_acc_z_" + group + ".txt",
    ]
    # body acceleration
    filenames += [
        "body_acc_x_" + group + ".txt",
        "body_acc_y_" + group + ".txt",
        "body_acc_z_" + group + ".txt",
    ]
    # body gyroscope
    filenames += [
        "body_gyro_x_" + group + ".txt",
        "body_gyro_y_" + group + ".txt",
        "body_gyro_z_" + group + ".txt",
    ]
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + "/y_" + group + ".txt")
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=""):
    # load all train
    trainX, trainy = load_dataset_group("train", prefix)

    # load all test
    testX, testy = load_dataset_group("test", prefix)

    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1

    return trainX, trainy.flatten(), testX, testy.flatten()


def HAR_data_generating():
    train_X, train_Y, test_X, test_Y = load_dataset("data/HAR/raw/")
    return np.dsplit(train_X, 9 // 3), train_Y, np.dsplit(test_X, 9 // 3), test_Y
