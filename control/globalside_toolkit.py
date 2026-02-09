import torch
import math
import json
from control.Enums import (
    ModelType,
    DatasetInfo,
    FLFramework,
    SpeiclaClientType,
    DatasetName,
    ParametersForDataSplitType,
)
from control.Client import Client
from models.LSTMAE import MMLSTMAE
from models.MLP import MLP, MLP1
from models.LSTMDIS import LSTMDIS,TSDiscriminator
from models.LSTMCLASSIFY import LSTMCLASSIFY
from tools.utils import (
    serialize_model,
    deserialize_model,
)
from typing import Union
from types import SimpleNamespace
import numpy as np
from control.Enums import (
    LearningType,
    FLFramework,
    generate_client_id,
    parse_client_id,
    ParametersForDataSplitType,
)
from control.Manager import HyperInfo
from control.MMULFED.ClientMMULFED import ClientMMULFED
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def create_clients(
    global_M_set: list,
    M_sets: list,
    client_nums_for_Dk: list,
    client_modality_choice_idx_in_SL: list,
    hyper_info: HyperInfo,
    fl_framework: str,
    aux_model_hyperinfo_list: list = None,
    is_global: bool = False,
):
    clients = {}
    for k in range(len(M_sets)):
        M_set = M_sets[k]
        for p in range(client_nums_for_Dk[k]):
            hyper_info = deepcopy(hyper_info)
            hyper_info.set_modaility_set(M_set)
            hyper_info.modality_choice_idx_in_SL = client_modality_choice_idx_in_SL[k]
            client_id = generate_client_id(k, p)
            if is_global:
                client_id = SpeiclaClientType.GLOBAL_CLIENT.value
            if fl_framework in (FLFramework.MMULFED.value, FLFramework.MMULFEDAA.value):
                clients[client_id] = ClientMMULFED(
                    hyper_info, client_id, deepcopy(aux_model_hyperinfo_list)
                )
            elif fl_framework == FLFramework.FEDAVG.value:
                clients[client_id] = Client(hyper_info, client_id)
            print("client", client_id, "is generated")
    return clients


def sample_clients_normal(client_keys: list, client_num: int):
    client_num1 = min(client_num, len(client_keys))
    indexes = np.random.choice(
        list(range(len(client_keys))), client_num1, replace=False
    )
    indexes.sort()
    selected_client_keys = [client_keys[idx] for idx in indexes]
    return selected_client_keys


def aggregate_model_paras_in_list(serialized_params_list: list, weights=None):
    if weights is None:
        weights = torch.ones(len(serialized_params_list))

    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)

    weights = weights / torch.sum(weights)
    # print("weights", weights)
    assert torch.all(weights >= 0), "weights should be non-negative values"
    serialized_parameters = torch.sum(
        torch.stack(serialized_params_list, dim=-1) * weights, dim=-1
    )

    return serialized_parameters


def aggregate_collected_encoders_decoders(
    encoder_list: list,
    model_params_cache_encoders: list,
    decoder_list: list,
    model_params_cache_decoders: list,
    global_M_set: Union[tuple, list],
):
    assert len(model_params_cache_encoders) == len(model_params_cache_decoders)
    # channel_num = len(model_params_cache_encoders)
    # print("aggregate channel_num:", channel_num)
    print("start to aggregate...")
    for global_idx, modality in enumerate(global_M_set):
        if len(model_params_cache_encoders[global_idx]) > 0:

            aggregated_encoder_params = aggregate_model_paras_in_list(
                model_params_cache_encoders[global_idx]
            )

            encoder = encoder_list[global_idx]

            deserialize_model(encoder, aggregated_encoder_params)
            print(f"deserialize m{modality}-" + encoder.name)

        if len(model_params_cache_decoders[global_idx]) > 0:
            aggregated_decoder_params = aggregate_model_paras_in_list(
                model_params_cache_decoders[global_idx]
            )
            decoder = decoder_list[global_idx]
            deserialize_model(decoder, aggregated_decoder_params)


def collect_encoders_decoders(
    encoder_list: list,
    model_params_cache_encoders: list,
    decoder_list: list,
    model_params_cache_decoders: list,
    client_M_set_encoders: Union[tuple, list],
    client_M_set_decoders: Union[tuple, list],
    global_M_set: Union[tuple, list],
):
    assert len(model_params_cache_encoders) == len(model_params_cache_decoders)
    assert len(model_params_cache_encoders) == len(global_M_set)
    # print("collect channel_num:", channel_num)

    """
    The encoder_list of a client only contains the encoders for its modalities and by default these
    encoders are thought to have been trained.
    """

    def get_submodel_serialized_parameters(submodel_list: list):
        submodel_paras = []
        print("start to collect", end=":")
        for i in range(len(submodel_list)):
            submodel: torch.nn.modules = submodel_list[i]
            submodel_paras.append(serialize_model(submodel))
            print(submodel.name, end="\t")
        return submodel_paras

    serialized_encoders = get_submodel_serialized_parameters(encoder_list)
    serialized_decoders = get_submodel_serialized_parameters(decoder_list)
    for local_idx, modality in enumerate(client_M_set_encoders):
        global_idx = global_M_set.index(modality)
        model_params_cache_encoders[global_idx].append(serialized_encoders[global_idx])
    for local_idx, modality in enumerate(client_M_set_decoders):
        global_idx = global_M_set.index(modality)
        model_params_cache_decoders[global_idx].append(serialized_decoders[global_idx])


def get_model(model_args, initial="n"):
    split_paras = ParametersForDataSplitType(model_args["data_split_type"])
    dataset_info = DatasetInfo(model_args["dataset_name"], split_paras.global_m_set)
    model_type = model_args["model_type"]
    if model_type == ModelType.LSTMAE.value:
        input_sizes = dataset_info.input_sizes
        use_act_list = dataset_info.use_act_list
        model = MMLSTMAE(
            input_size=model_args["input_size_LSTMAE"],
            hidden_size=model_args["hidden_size_LSTMAE"],
            dropout_ratio=model_args["dropout"],
            training_mode=model_args["training_mode"],
            channel_num=model_args["channel_num"],
            input_sizes=input_sizes,
            use_act_list=use_act_list,
            name=f"global;{model_type}",
            encoder_only=False,
        )
        model.to(device)
        return model
    elif model_type == ModelType.MLP.value:
        hidden_size2 = 36
        if model_args["dataset_name"] == DatasetName.UCF101.value:
            hidden_size2 = 108
        model = MLP(
            input_size=model_args["input_size_MLP"],
            hidden_size=model_args["hidden_size_MLP"],
            output_size=model_args["output_size_MLP"],
            hidden_size2=hidden_size2,
            init=initial,
        )
        model.to(device)
        return model
    elif model_type == ModelType.MLP1.value:
        hidden_size2 = 36
        if model_args["dataset_name"] == DatasetName.UCF101.value:
            hidden_size2 = 108
        model = MLP1(
            input_size=model_args["input_size_MLP"],
            output_size=model_args["output_size_MLP"],
            init=initial,
        )
        model.to(device)
        return model
    elif model_type == ModelType.LSTMDIS.value:
        # model = LSTMDIS(
        #     input_size=model_args["input_size"], hidden_size=model_args["hidden_size"]
        # )
        model = TSDiscriminator(model_args["input_size"])
        model.to(device)
        return model
    elif model_type == ModelType.LSTMCLASSIFY.value:
        model = LSTMCLASSIFY(
            input_size=model_args["input_size_LSTMAE"],
            hidden_size=model_args["hidden_size_LSTMAE"],
            fc_dim=256,
            lstm_layer_num=3,
            num_classes=49,
        )
        model.to(device)
        return model


def get_local_model(local_model_path):
    # decompose model name for model L

    local_model_args = None
    with open(local_model_path + "/config.json", "r", encoding="utf-8") as f:
        args_dict = json.load(f)
        local_model_args = SimpleNamespace(**args_dict)

    local_model_args1 = vars(local_model_args)
    model = get_model(local_model_args1)
    model.load_state_dict(
        torch.load(local_model_path + "/" + local_model_path.split("/")[-1] + ".pt")
    )

    return model, local_model_args


def make_client_group(client_keys: list):
    client_group_dict = {}
    for k in client_keys:
        modality_type, number_under_type = parse_client_id(k)
        if modality_type not in client_group_dict:
            client_group_dict[modality_type] = [k]
        else:
            client_group_dict[modality_type].append(k)
    for k in client_group_dict:
        client_group_dict[k].sort()

    return client_group_dict
