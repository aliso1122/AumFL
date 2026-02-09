import torch

from control.Enums import InTrainingMode, DataAnalysisDimensions, DataAnalysisIndicators
import math


def cosine_decay_lr(lr: float, t: int, T: int):
    return 0.5 * (1 + math.cos(t * math.pi / T)) * lr


def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    """Unfold model parameters

    Unfold every layer of model, concate all of tensors into one.
    Return a `torch.Tensor` with shape (size, ).

    Args:
        model (torch.nn.Module): model to serialize.
    """

    parameters = [param.data.view(-1) for param in model.parameters()]
    m_parameters = torch.cat(parameters)
    m_parameters = m_parameters.cpu()

    return m_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor, mode="copy"
):
    """Assigns serialized parameters to model.parameters.
    This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
    NOTE: this function manipulates ``model.parameters``.

    Args:
        model (torch.nn.Module): model to deserialize.
        serialized_parameters (torch.Tensor): serialized model parameters.
        mode (str): deserialize mode. "copy" or "add".
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        if mode == "copy":
            parameter.data.copy_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        elif mode == "add":
            parameter.data.add_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        else:
            raise ValueError(
                'Invalid deserialize mode {}, require "copy" or "add" '.format(mode)
            )
        current_index += numel


def get_log_info(
    log_type: str,
    global_epoch: int = None,
    client_id: int = None,
    in_training_metrics: tuple = None,
    metrics_name: tuple = None,
    model_name: str = None,
    extra: str = None,
):
    log_info = {DataAnalysisDimensions.LOG_TYPE.value: log_type}
    assert len(in_training_metrics) == len(metrics_name)
    log_info["in_training_metrics"] = in_training_metrics
    log_info["metrics_name"] = metrics_name
    log_info[DataAnalysisDimensions.GLOBAL_EPOCH.value] = global_epoch
    log_info[DataAnalysisDimensions.ClIENT_ID.value] = client_id
    log_info[DataAnalysisDimensions.Model_Name.value] = model_name
    log_info["extra"] = extra
    return log_info


def log_flatten(log_info):
    datatype_dict = {
        "str": "basic",
        "tuple": "listlike",
        "int": "basic",
        "float": "basic",
        "list": "listlike",
    }

    data_matrix = [[]]
    column_name = []
    if log_info[DataAnalysisDimensions.LOG_TYPE.value] in (
        InTrainingMode.TRAIN.value,
        InTrainingMode.TEST.value,
    ):

        column_name = (
            [e.value for e in DataAnalysisDimensions.__members__.values()]
            + [e.value for e in DataAnalysisIndicators.__members__.values()]
            + ["flatten_id"]
        )
        data_matrix[0] = [None] * (len(column_name) - 1) + [0]
        for e in DataAnalysisDimensions.__members__.values():
            data_matrix[0][column_name.index(e.value)] = log_info[e.value]

        assert len(log_info["in_training_metrics"]) == len(log_info["metrics_name"])
        for metric_idx in range(len(log_info["in_training_metrics"])):
            metric_name = log_info["metrics_name"][metric_idx]
            metric = log_info["in_training_metrics"][metric_idx]
            datatype = str(type(metric))[8:-2]
            col_index = column_name.index(metric_name)
            if datatype_dict[datatype] == "basic":
                data_matrix[0][col_index] = metric
            elif datatype_dict[datatype] == "listlike":
                data_matrix[0][col_index] = metric[0]
                for i in range(1, len(metric)):
                    if len(data_matrix) - 1 >= i:
                        data_matrix[i][col_index] = metric[i]
                    else:
                        new_row = [None] * len(data_matrix[0])
                        new_row[-1] = i  # set flatten_id
                        new_row[col_index] = metric[i]
                        data_matrix.append(new_row)
        for row_id in range(1, len(data_matrix)):
            row = data_matrix[row_id]
            for i in range(len(row)):
                if row[i] is None:
                    row[i] = data_matrix[0][i]

    return data_matrix, column_name


def convert_model_info_to_matrix(model_infos):
    ori_data = []
    col_name_list = []
    
    for item in model_infos:
        if str(type(item))[8:-2] == "str":
            continue
        data_matrix, col_names = log_flatten(item)
        if len(col_names) == 0:
            continue
        else:
            col_name_list = col_names
        for row in data_matrix:
            ori_data.append(row)
    return ori_data, col_name_list
