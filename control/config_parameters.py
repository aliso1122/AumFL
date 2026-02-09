import argparse
from control.Enums import (
    LearningType,
    ModelType,
    DatasetName,
    PurposeType,
    TrainingMode4LSTMAE,
    ModelLoading,
    DataSplitType,
    LearningRateDecay,
    ParametersForDataSplitType,
    FLFramework,
    FLSubType,
)
from types import SimpleNamespace
import json
import os

PARAMETERS = {
    "purpose": {
        "type": str,
        "default": PurposeType.TRAIN.value,
        "choices": [e.value for e in PurposeType.__members__.values()],
        "description": "purpose of the main function; see details in PurposeType definition",
        "parents": ["/"],
    },
    "fl_framework": {
        "type": str,
        "default": FLFramework.FEDAVG.value,
        "choices": [e.value for e in FLFramework.__members__.values()],
        "description": "federated learning framework",
        "parents": ["/"],
    },
    "fl_subtype": {
        "type": str,
        "default": "",
        "choices": [e.value for e in FLSubType.__members__.values()],
        "description": "federated learning framework(subtype)",
        "parents": ["/fl_framework:{}/".format(FLFramework.FEDAVG.value)],
    },
    "model_id": {
        "type": str,
        "default": "",
        "description": "the sole identification of a model",
        "parents": ["/"],
    },
    "model_dir": {
        "type": str,
        "default": "trained_models",
        "description": "the directory to store trained models",
        "parents": ["/"],
    },
    "learning_type": {
        "type": str,
        "default": LearningType.UNSUPERVISED.value,
        "choices": [e.value for e in LearningType.__members__.values()],
        "description": "decides the method to train models",
        "parents": ["/"],
    },
    "model_type": {
        "type": str,
        "default": ModelType.LSTMAE.value,
        "choices": [e.value for e in ModelType.__members__.values()],
        "description": "decides what model is trained",
        "parents": ["/"],
    },
    "dataset_name": {
        "type": str,
        "default": DatasetName.HAR.value,
        "choices": [e.value for e in DatasetName.__members__.values()],
        "parents": ["/"],
    },
    "data_split_type": {
        "type": str,
        "default": DataSplitType.TYPE1.value,
        "choices": [e.value for e in DataSplitType.__members__.values()],
        "description": "decides how to split the data into sub-datasets; see details in DataSplitType definition",
        "parents": ["/"],
    },
    "batch_size": {
        "type": int,
        "default": 16,
        "parents": ["/"],
    },
    "learning_rate": {
        "type": float,
        "default": 1e-5,
        "parents": ["/"],
    },
    "learning_rate_decay": {
        "type": str,
        "default": LearningRateDecay.NONE.value,
        "choices": [e.value for e in LearningRateDecay.__members__.values()],
        "parents": ["/"],
    },
    "global_epoch_number": {
        "type": int,
        "default": 100,
        "parents": ["/"],
    },
    "local_epoch_number": {
        "type": int,
        "default": 3,
        "parents": ["/"],
    },
    "log_interval": {
        "type": int,
        "default": 50,
        "description": "how many batch iteration as an interval to log status",
        "parents": ["/"],
    },
    "load_model": {
        "type": int,
        "default": ModelLoading.LOADNOTHING.value,
        "choices": [e.value for e in ModelLoading.__members__.values()],
        "parents": ["/"],
    },
    "load_model_name": {
        "type": str,
        "default": "",
        "description": "the name of the model to be loaded",
        "parents": ["/load_model:{}/".format(ModelLoading.LOADMODEL.value)],
    },
    "load_aux_model": {
        "type": int,
        "default": ModelLoading.LOADNOTHING.value,
        "choices": [e.value for e in ModelLoading.__members__.values()],
        "parents": ["/"],
    },
    "load_aux_model_name": {
        "type": str,
        "default": "",
        "description": "the name of the aux model to be loaded",
        "parents": ["/load_aux_model:{}/".format(ModelLoading.LOADMODEL.value)],
    },
    "pretrained_model_id": {
        "type": str,
        "default": "",
        "description": "the id of the pretrained model trained stored in the local project",
        "parents": ["/learning_type:{}/".format(
            LearningType.SUPERVISED_WITH_ENCODER.value
        )],
    },
    "with_aux": {
        "type": int,
        "default": 0,
        "description": "",
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "AC_gap": {
        "type": int,
        "default": 1,
        "description": "",
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "AC_local_epoch": {
        "type": int,
        "default": 0,
        "description": "",
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "alternative_alignment": {
        "type": int,
        "default": 0,
        "description": "",
        "parents": ["/fl_framework:{}/".format(FLFramework.FEDAVG.value)],
    },
    "client_num_per_epoch": {
        "type": int,
        "default": -1,
        "description": "the num of selected clients for one global traininig epoch",
        "parents": ["/"],
    },
    # for MLP
    "hidden_size_MLP": {
        "type": int,
        "default": 64,
        "parents": ["/model_type:{}/".format(ModelType.MLP.value)],
    },
    "input_size_MLP": {
        "type": int,
        "default": 64,
        "parents": ["/model_type:{}/".format(ModelType.MLP.value)],
    },
    "output_size_MLP": {
        "type": int,
        "default": 64,
        "parents": ["/model_type:{}/".format(ModelType.MLP.value)],
    },
    # for LSTM
    "input_size_LSTMAE": {
        "type": int,
        "default": 3,
        "choices": [3],
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
    },
    "hidden_size_LSTMAE": {
        "type": int,
        "default": 256,
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
    },
    "channel_num": {
        "type": int,
        "default": 3,
        "choices": [1, 2, 3],
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
        "description": "the id of the pretrained model trained stored in the local project",
    },
    "dropout": {
        "type": float,
        "default": 0,
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
        "description": "dropout ratio",
    },
    "seq_len": {
        "type": int,
        "default": 128,
        "choices": [128],
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
        "description": "the length of the input sequence of LSTM",
    },
    "training_mode": {
        "type": str,
        "default": TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
        "choices": [e.value for e in TrainingMode4LSTMAE.__members__.values()],
        "parents": ["/model_type:{}/".format(ModelType.LSTMAE.value)],
        "description": "mode of training; see details in LSTM",
    },
    #for alignment-aware(aa) fed
    "aa_dropout_eps1": {
        "type": float,
        "default": 0.0001,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_pvalue1": {
        "type": float,
        "default": 0.05,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_eps2": {
        "type": float,
        "default": 0.0001,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_pvalue2": {
        "type": float,
        "default": 0.05,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_L": {
        "type": float,
        "default": 0.1,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_T_imp": {
        "type": int,
        "default": 100,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "aa_dropout_frozen_length": {
        "type": int,
        "default": 10,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
    "dis_threshold": {
        "type": float,
        "default": 0.6,
        "parents": ["/fl_framework:{}/".format(FLFramework.MMULFEDAA.value)],
    },
}


def is_useful_para(k: str, args_dict: dict):
    parents = PARAMETERS[k]["parents"]
    def belong_to_parent(parent):
        if parent == "/":
            return True
        parent_list = parent.split("/")
        flag = True
        for info in parent_list:
            if info == "":
                continue
            else:
                parent_k, parent_v = info.split(":")
                if str(args_dict[parent_k]) != str(parent_v):
                    flag = False
                    break
        return flag
    
    for parent_path in parents:
        if belong_to_parent(parent_path):
            return True
    return False


def extract_useful_paras(args_dict):
    args = {}
    for k in args_dict:
        if is_useful_para(k, args_dict):
            args[k] = args_dict[k]
        else:
            args[k] = None
    return args


def args_check(args_dict):
    parameters = ParametersForDataSplitType(args_dict["data_split_type"])
    args_dict["channel_num"] = len(parameters.global_m_set)
    total_client_num = sum(parameters.client_nums_for_Dk)
    if args_dict["client_num_per_epoch"] <= 0:
        args_dict["client_num_per_epoch"] = total_client_num
    elif args_dict["client_num_per_epoch"] > total_client_num:
        print(
            "warining: client_num_per_epoch is greater than total_client_num and so it is set to total_client_num"
        )
        args_dict["client_num_per_epoch"] = total_client_num

    assert not (args_dict["learning_type"] in (LearningType.SUPERVISED_FOR_AUX.value, LearningType.UNSUPERVISED_WITH_AUX.value))
    assert not (args_dict["learning_type"] != LearningType.UNSUPERVISED.value and args_dict["fl_framework"] == FLFramework.MMULFED.value)

def args_complete(args_dict):
    for k in PARAMETERS:
        if k not in args_dict and is_useful_para(k, args_dict):
            args_dict[k] = PARAMETERS[k]["default"]
            print(f"Warining: para {k} is added by default")


def get_config(args):
    level1_dir = args.model_dir
    if not os.path.exists(level1_dir):
        os.makedirs(level1_dir)
    level2_dir = level1_dir + "/" + args.dataset_name
    if not os.path.exists(level2_dir):
        os.makedirs(level2_dir)

    level3_dir = level2_dir + "/" + args.data_split_type
    if not os.path.exists(level3_dir):
        os.makedirs(level3_dir)

    leaf_dir = level3_dir + "/" + args.model_id
    if not os.path.exists(leaf_dir):
        os.mkdir(leaf_dir)

    config_path = leaf_dir + "/" + "config.json"
    args_dict = vars(args)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            print("has loaded the existing config from " + config_path)
            args_dict = json.load(f)
            args_dict["purpose"] = args.purpose
            args_complete(args_dict)
    else:
        args_dict = extract_useful_paras(args_dict)

    args_dict["model_level_dir"] = leaf_dir
    args_check(args_dict)
    with open(config_path, "w", encoding="utf-8") as f:
        print("has stored the config")
        purpose = args_dict["purpose"]
        del args_dict["purpose"]
        json.dump(args_dict, f, indent=4)
        args_dict["purpose"] = purpose
    print(args_dict)
    return SimpleNamespace(**args_dict)


def get_args():
    parser = argparse.ArgumentParser(description="")
    for key in PARAMETERS:
        value = PARAMETERS[key]
        help_tmp = None
        choices_tmp = None
        if help_tmp in value:
            help_tmp = value["description"]
        if choices_tmp in value:
            choices_tmp = choices_tmp["choices"]
        parser.add_argument(
            "--" + key,
            type=value["type"],
            default=value["default"],
            choices=choices_tmp,
            help=help_tmp,
        )

    args = parser.parse_args()

    return args
