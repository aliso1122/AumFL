from typing import Union
import torch
from torch import nn
import numpy as np
from models.LSTMAE import MMLSTMAE, Encoder
import math
from control.Enums import DatasetInfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def z_norm(x: torch.Tensor, dim:int):
    return torch.nan_to_num(nn.functional.normalize(x, dim=dim))
def data_preprocess_transposed(
    data: Union[tuple, list], trans: int = 0, flatten_targets: int = 0
):
    """
    orignial data is (modality1, modality2, .... , modalityN, target)
    preprocessed data is ((modality1, modality2, .... , modalityN), target)
    """
    data_x = []
    for id, data_ in enumerate(data, 0):
        if id == len(data) - 1:
            labels = data_
            if flatten_targets == 1:
                labels = labels.flatten()
            return data_x, labels
        if trans == 1:
            data_x.append(z_norm(torch.transpose(data_, 0, 1), 1).to(device))
        elif trans == 0:
            data_x.append(z_norm(data_,1).to(device))
        else:
            assert False


def multimodal_fusion(data_list):
    # for a in data_list:
    #     print(a.shape)
    return torch.concatenate(data_list, dim=-1)


def extract_feature_from_HAR_raw(
    encoder: Encoder, data_: tuple, modalities: list = None
):
    output = []
    for i in range(len(data_)):
        data: torch.Tensor = data_[i]
        encoder.set_encoder_only()
        model_out: torch.Tensor = encoder(data).to(device)
        encoder.setback()
        # (1, batch_size, hidden_size)->(batch_size, hidden_size)
        model_out = model_out.squeeze()
        output.append(model_out)
    return output


def encode_data_LSTMAE(
    encoder: MMLSTMAE,
    data_: tuple,
    local_modality_set: list,
    global_modality_set: list,
    dataset_info: DatasetInfo,
):
    output = []
    if dataset_info.use_time_step_dim == 0:
        for local_idx, modality in enumerate(local_modality_set):
            data: torch.Tensor = data_[local_idx]
            global_idx = global_modality_set.index(modality)
            encoder_ = encoder.encoder_list[global_idx]
            encoder.set_encoder_only()
            model_out: torch.Tensor = encoder_(data).to(device)
            encoder.setback()
            # (1, batch_size, hidden_size)->(batch_size, hidden_size)
            model_out = model_out.squeeze(0)
            output.append(model_out)
    elif dataset_info.use_time_step_dim == 1:
        for local_idx, modality in enumerate(local_modality_set):
            data: torch.Tensor = data_[local_idx]
            global_idx = global_modality_set.index(modality)
            encoder_ = encoder.encoder_list[global_idx]
            encoder.setback()
            # model_out (batch_size, seq_len, hidden_size)

            model_out: torch.Tensor = encoder_(data).to(device)
            if dataset_info.batch_dimension_switched == 1:
                model_out = torch.transpose(model_out, dim0=1, dim1=0)

            model_out = model_out.reshape(
                (model_out.shape[0] * model_out.shape[1], model_out.shape[2])
            )
            # print(model_out.shape)
            output.append(model_out)
    return output
