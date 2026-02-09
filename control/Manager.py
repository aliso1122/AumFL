import torch.nn as nn
import time
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from typing import Union
from control.Enums import (
    LearningType,
    DatasetName,
    ModelType,
    InTrainingMode,
    LearningRateDecay,
    SpeiclaClientType,
    DatasetInfo,
    FLSubType,
)
import random
from data_generating.utils import sequence_dataset
from torchmetrics import F1Score
from sklearn.metrics import f1_score
from types import SimpleNamespace
from control.localtraning_toolkit import (
    encode_data_LSTMAE,
    multimodal_fusion,
    data_preprocess_transposed,
    extract_feature_from_HAR_raw,
)
from tools.utils import cosine_decay_lr
from copy import deepcopy
from tools.plot_tools import plot_orig_vs_reconstructed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class HyperInfo:
    def __init__(
        self,
        learning_type: str,
        modality_choice_idx_in_SL: int,
        model_type: str,
        dataset_name: str,
        data_split_type: str,
        data_preprocess_method: str,
        global_modaility_set: tuple,
        modaility_set: tuple,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        learning_rate: float,
        learning_rate_decay: str,
        encoder: nn.Module,
        encoder_info: SimpleNamespace,
        alternative_alignment: int = 0,
        fl_subtype: str = "",
    ):
        self.learning_type = learning_type
        self.modality_choice_idx_in_SL = modality_choice_idx_in_SL
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.data_split_type = data_split_type
        self.data_preprocess_method = None
        self.global_modaility_set = global_modaility_set
        self.modaility_set = modaility_set
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.encoder = encoder
        self.encoder_info = encoder_info
        self.learning_rate_decay = learning_rate_decay
        self.comp_modality_set = []
        self.alternative_alignment = alternative_alignment
        self.fl_subtype = fl_subtype

    def set_modaility_set(self, m_set):
        self.modaility_set = m_set
        self.comp_modality_set = []
        for m in self.global_modaility_set:
            if m not in self.modaility_set:
                self.comp_modality_set.append(m)


class Manager:
    """
    Managers implement training a certain model under a certain dataset.
    A manager requires a model type and a dataset to be specified when it is created. Accordingly,
    it registers correpsonding functions for training process, including data preprocessing, data forwarding
    lossing computing, gradient calculating and model updating.
    """

    def __init__(self, hyperInfo: HyperInfo, owner: str = ""):

        self.owner = owner
        self.model_type = hyperInfo.model_type
        self.fl_subtype = hyperInfo.fl_subtype
        self.reserved_global_model = None
        self.reserved_global_control_variate = None
        self.dataset_name = hyperInfo.dataset_name
        self.learning_type = hyperInfo.learning_type
        self.modality_choice_in_SL = hyperInfo.modality_choice_idx_in_SL

        self.encoder = None
        self.encoder_info = None
        self.forward_compute = None
        self.loss_compute = None

        if self.learning_type == LearningType.UNSUPERVISED.value:
            self.alternavtie_alignment = hyperInfo.alternative_alignment
            if all(
                [
                    self.model_type == ModelType.LSTMAE.value,
                    self.dataset_name == DatasetName.HAR.value,
                ]
            ):
                # if self.owner == "1_4_":
                #     assert False, "hihihi"
                self.forward_compute = self.__forward_compute_LSTMAE_UL
                self.loss_compute = self.__loss_compute_LSTMAE_UL

        elif (
            self.learning_type == LearningType.SUPERVISED_WITH_ENCODER.value
            or self.learning_type == LearningType.SUPERVISED_FOR_AUX.value
        ):
            self.loss_compute = self.__loss_compute_SL
            if self.learning_type == LearningType.SUPERVISED_WITH_ENCODER.value:
                self.encoder_info = hyperInfo.encoder_info
                self.encoder = hyperInfo.encoder
                self.encoder.requires_grad_(False)  # no tunning
                if all(
                    [
                        self.encoder_info.model_type == ModelType.LSTMAE.value,
                        self.dataset_name == DatasetName.HAR.value,
                        self.learning_type
                        == LearningType.SUPERVISED_WITH_ENCODER.value,
                    ]
                ):
                    self.forward_compute = self.__forward_compute_SL_LSTMAE_Encoded
            elif all(
                [
                    self.dataset_name == DatasetName.HAR.value,
                    self.learning_type == LearningType.SUPERVISED_FOR_AUX.value,
                ]
            ):
                self.forward_compute = self.__forward_compute_SL4AUX

        self.dataset_info = DatasetInfo(self.dataset_name)
        self.to_transpose = self.dataset_info.batch_dimension_switched

        self.modaility_set = hyperInfo.modaility_set
        self.global_modaility_set = hyperInfo.global_modaility_set
        self.comp_modality_set = hyperInfo.comp_modality_set
        self.model: nn.Module = hyperInfo.model
        self.local_control_variate = None
        if self.fl_subtype == FLSubType.SCAFFOLD.value:
            self.local_control_variate = deepcopy(self.model)
        self.criterion = hyperInfo.criterion
        self.learning_rate = hyperInfo.learning_rate
        self.learning_rate_decay = hyperInfo.learning_rate_decay
        self.train_dataloader = hyperInfo.train_dataloader
        self.val_dataloader = hyperInfo.val_dataloader
        self.test_dataloader = hyperInfo.test_dataloader

        if self.owner == SpeiclaClientType.GLOBAL_CLIENT.value:
            assert self.train_dataloader.dataset.M_set == self.modaility_set
        else:
            assert (
                self.train_dataloader.dataset.M_set == self.val_dataloader.dataset.M_set
                and self.val_dataloader.dataset.M_set
                == self.test_dataloader.dataset.M_set
                and self.test_dataloader.dataset.M_set == self.modaility_set
            )

        self.ae_aux_model: nn.Module = None
        self.generator: nn.Module = None  # for training the aux
        self.criterion_ul_aux = torch.nn.CrossEntropyLoss()

    def reset_environment(self, new_learning_type: str):
        if new_learning_type in [
            LearningType.UNSUPERVISED_AB.value,
            LearningType.UNSUPERVISED.value,
            LearningType.UNSUPERVISED_WITH_AUX.value,
        ]:
            self.learning_type = new_learning_type
        else:
            assert False

    def __update_model_encoder_only(self, loss, learning_rate):
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=self.model.parameters(),
            allow_unused=True,
        )
        for name_and_param, grad in zip(self.model.named_parameters(), grads):
            name, param = name_and_param
            if grad is not None and name.find("encoder") != -1:
                param.data.sub_(learning_rate * grad)
            else:
                param.data.sub_(0)

    def update_control_variate(self, lr, k):
        for name_and_param, name_and_param2, name_and_param3, name_and_param4 in zip(
            self.local_control_variate.named_parameters(),
            self.reserved_global_control_variate.named_parameters(),
            self.model.named_parameters(),
            self.reserved_global_model.named_parameters(),
        ):
            _, local_control = name_and_param
            local_control_1 = deepcopy(local_control)
            _, global_control = name_and_param2
            _, local_model_para = name_and_param3
            _, global_model_para = name_and_param4
            local_control.sub_(
                local_control_1
                - global_control
                + (global_model_para - local_model_para) / (lr * k)
            )

    def update_model(self, loss, learning_rate):
        # print("here", loss.dtype)
        grads = torch.autograd.grad(
            outputs=loss, inputs=self.model.parameters(), allow_unused=True
        )
        if self.fl_subtype == FLSubType.FEDPROX.value:
            for name_and_param, grad, name_and_param2 in zip(
                self.model.named_parameters(),
                grads,
                self.reserved_global_model.named_parameters(),
            ):
                name, param = name_and_param
                name2, param2 = name_and_param2
                if grad is None:
                    param.data.sub_(0)
                else:
                    param.data.sub_(learning_rate * grad + 0.01 * (param - param2))
        elif self.fl_subtype == FLSubType.FEDPROX.value:
            for name_and_param, grad, name_and_param2, name_and_param3 in zip(
                self.model.named_parameters(),
                grads,
                self.reserved_global_control_variate.named_parameters(),
                self.local_control_variate.named_parameters(),
            ):
                name, param = name_and_param
                name2, param2 = name_and_param2
                name3, param3 = name_and_param3
                if grad is None:
                    param.data.sub_(0)
                else:
                    param.data.sub_(
                        learning_rate * grad + learning_rate * (param2 - param3)
                    )
        else:
            for name_and_param, grad in zip(self.model.named_parameters(), grads):
                name, param = name_and_param
                # if name == "fc.fc3.bias":
                # print(name + "-para", param)
                # print(name + "-grad", grad)
                if grad is None:
                    param.data.sub_(0)
                else:
                    param.data.sub_(learning_rate * grad)
                # print(name, "grad", grad.abs().mean().item())

        # assert False

    def __forward_compute_LSTMAE_UL(self, data_: tuple, modality: int = 0):
        """
        The returned is a tuple containing real outputs from LSTMAE's decoders.
        """
        ori_local_idx = self.modaility_set.index(modality)
        ori_global_idx = self.global_modaility_set.index(modality)
        data: torch.Tensor = data_[ori_local_idx]
        model_out = self.model(data, ori_global_idx)
        final_out = []
        for m in self.modaility_set:
            des_global_idx = self.global_modaility_set.index(m)
            final_out.append(model_out[des_global_idx])
        return tuple(final_out)

    def __forward_compute_LSTMAE_UL_with_aux(self, data: tuple, ori: int, des: int):
        ori_local_idx = self.modaility_set.index(ori)
        ori_global_idx = self.global_modaility_set.index(ori)
        des_global_idx = self.global_modaility_set.index(des)
        ori_data_x = data[ori_local_idx]

        model_out = self.ae_aux_model(
            ori_data_x,
            ori_global_idx,
            des_global_idx,
            des_global_idx,
            self.to_transpose,
        )

        return model_out

    def __forward_compute_LSTMAE_UL_AB(self, data: tuple, ori: int, des: int):
        """
        The returned is the decoded b-modality data from encoded a-modality data.
        """

        ori_local_idx = self.modaility_set.index(ori)
        ori_global_idx = self.global_modaility_set.index(ori)
        des_global_idx = self.global_modaility_set.index(des)

        input_data: torch.Tensor = data[ori_local_idx]
        model_out = self.model(input_data, ori_global_idx)[des_global_idx]
        return model_out

    def __forward_compute_SL_LSTMAE_Encoded(self, data: tuple):
        encoded = encode_data_LSTMAE(
            self.encoder,
            data,
            self.modaility_set,
            self.global_modaility_set,
            self.dataset_info,
        )
        input = multimodal_fusion(encoded)

        out: torch.Tensor = self.model(input).to(device)

        return out

    def __forward_compute_SL_LSTMAE_Encoded2(self, data: tuple, target: torch.Tensor):
        encoded = encode_data_LSTMAE(
            self.encoder,
            data,
            self.modaility_set,
            self.global_modaility_set,
            self.dataset_info,
        )
        input = torch.concatenate(encoded, dim=0)
        out: torch.Tensor = self.model(input).to(device)
        targets = torch.concatenate([target, target.clone()], dim=0)

        return out, targets

    def __forward_compute_SL_LSTMAE_Encoded3(self, data: torch.Tensor):

        encoder_ = self.encoder.encoder_list[0]
        self.encoder.set_encoder_only()
        with torch.no_grad():
            encoded: torch.Tensor = encoder_(data).to(device).squeeze(0)
        self.encoder.setback()

        out: torch.Tensor = self.model(encoded).to(device)
        return out

    def __forward_compute_SL4AUX(self, data: tuple, ori: int = 0, des: int = 0, batch_id: int = -1):

        ori_global_idx = self.global_modaility_set.index(ori)
        des_global_idx = self.global_modaility_set.index(des)
        # print("hihi", self.to_transpose)
        fake = torch.tensor([]).to(device)
        if ori in self.modaility_set:
            ori_local_idx = self.modaility_set.index(ori)
            fake = self.generator(data[ori_local_idx], ori_global_idx)[des_global_idx]
            if self.to_transpose == 1:
                fake = torch.transpose(fake, 0, 1).to(device)

        real = torch.tensor([]).to(device)
        if des in self.modaility_set:
            des_local_idx = self.modaility_set.index(des)
            real = data[des_local_idx].to(device)
            if self.to_transpose == 1:
                real = torch.transpose(real, 0, 1)

        input_for_aux = torch.concatenate([fake, real], dim=0)
        target_for_aux = torch.concatenate(
            [
                torch.zeros([len(fake)], dtype=torch.int64),
                torch.ones([len(real)], dtype=torch.int64),
            ],
            dim=0,
        ).to(device)

        assert len(input_for_aux) == len(target_for_aux)
        new_order = torch.randperm(len(input_for_aux))
        input_for_aux = input_for_aux[new_order]
        target_for_aux = target_for_aux[new_order]
        # print(input_for_aux.shape, self.to_transpose == 1)
        if self.to_transpose == 1:
            input_for_aux = torch.transpose(input_for_aux, 0, 1)
   
       

        out: torch.Tensor = self.model(input_for_aux).to(device)

        probs = nn.functional.softmax(out, dim=1)  # (n, 2)
        selected_probs = probs.gather(1, target_for_aux.view(-1, 1))  # (n, 1)
        mean_probs = float(selected_probs.squeeze(1).mean())

        # print(out.shape, target_for_aux.shape)
        # if batch_id == 0:
        #     print("outout", out)
        return out, target_for_aux, mean_probs

    def __loss_compute_LSTMAE_UL(
        self,
        model_output: Union[tuple, list],
        target: Union[tuple, list],
        weights: Union[tuple, list] = (1,),
        mode: str = "",
    ):
        """
        If only 1 modality is involved, then output should be (modality1,) and target should be (target1,).
        Otherwise, output should be (modality1, modality2...) and target should be (target1, target2...).
        All parameters should have the same len.
        """
        # if mode == InTrainingMode.TEST.value:
        #     print(len(model_output), weights)
        #     # assert False
        assert len(model_output) == len(weights)
        assert len(model_output) == len(target)
        loss = weights[0] * self.criterion(model_output[0], target[0])
        for i in range(1, len(model_output)):
            loss = loss + weights[i] * self.criterion(model_output[i], target[i])
        # print(self.criterion)
        return loss

    def __loss_compute_SL(
        self, model_output: torch.Tensor, target: torch.Tensor, weights: list = None, target_long: bool = True
    ):
        if target_long:
            target = target.long()
        loss = self.criterion(model_output, target)
        return loss

    def __loss_compute_UL_with_AUX(
        self, model_output: torch.Tensor, target: torch.Tensor, weights: list = None
    ):
        loss = self.criterion_ul_aux(model_output, target)
        return loss

    def __get_ori_des(self, extra_info: dict):
        ori = extra_info["a"]
        des = extra_info["b"]
        return ori, des

    # def z_norm(self, x: torch.Tensor):
    #     return torch.nan_to_num(nn.functional.normalize(x, dim=1))

    def train_model_with_data_batches(
        self,
        mode: str,
        batch_id_left: int = -1,
        batch_id_right: int = -1,
        global_epoch: int = 0,
        local_epoch: int = 0,
        log_interval: int = 10,
        lr: float = 0,
        extra_info: dict = {},
        selected_pairs:list = []
    ):
        assert (batch_id_left >= 0 and batch_id_right >= 0) or (
            batch_id_left == -1 and batch_id_right == -1
        )
        assert (batch_id_left < batch_id_right) or (
            batch_id_left == -1 and batch_id_right == -1
        )
        loss_sum = 0
        acc_sum = 0
        f1_score = 0
        mae = 0
        f1 = F1Score(task="multiclass", num_classes=6).to(device)
        average_loss = 0
        average_acc = 0
        dataloader = None
        if mode == InTrainingMode.TRAIN.value:
            self.model.requires_grad_(True)
            # self.model.requires_grad_(True)
            dataloader = self.train_dataloader
            # print(len(self.train_dataloader))
            # assert False
        elif mode == InTrainingMode.EVALUATE.value:
            self.model.requires_grad_(False)
            dataloader = self.val_dataloader
        elif mode == InTrainingMode.EVALUATE_WITH_TRAINING_DATA.value:
            self.model.requires_grad_(False)
            dataloader = self.train_dataloader
        elif mode == InTrainingMode.TEST.value:
            self.model.requires_grad_(False)
            dataloader = self.test_dataloader
            # dataloader = self.train_dataloader

        # if self.dataset_info.use_time_step_dim == 1:
        # batch_size = dataloader.batch_size
        # dataloader = DataLoader(
        #     sequence_dataset(
        #         dataloader.dataset,
        #         (
        #             30  # random.randint(10, 40)
        #             if mode == InTrainingMode.TRAIN.value
        #             else 30
        #         ),
        #     ),
        #     batch_size,
        #     shuffle=True,
        # )

        pred = None
        model_out = None
        num_samples_iter = 0
        pred_record = torch.tensor([]).to(device)
        target_record = torch.tensor([]).to(device)
        for batch_idx, data_ in enumerate(dataloader, 0):
            if batch_id_left >= 0 and batch_id_right >= 0 and batch_idx < batch_id_left:
                continue

            # data is by default thought to be (modality1, modality2, .... , modalityN, target)
            # The first dimension of data is batch_size
            if batch_id_left > 0 and batch_id_right > 0:
                print("BATCH_IDX", batch_idx)
            data = None
            label = None
            target = None
            num_samples_iter += (
                len(self.modaility_set) * len(data_[0])
                if self.learning_type == LearningType.SUPERVISED_FOR_AUX.value
                else len(data_[0])
            )

            data, label = data_preprocess_transposed(
                data_, self.to_transpose, self.dataset_info.use_time_step_dim
            )
    
            # data = [self.z_norm(x) for x in data]

            if self.learning_type == LearningType.UNSUPERVISED.value:
                if self.fl_subtype == FLSubType.FEDUM.value:
                    loss = None
                    for local_idx, modality in enumerate(self.modaility_set):
                        model_out = [
                            self.__forward_compute_LSTMAE_UL_AB(
                                data, modality, modality
                            )
                        ]
                        target = [data[local_idx]]
                        if loss is None:
                            loss = self.__loss_compute_LSTMAE_UL(
                                model_output=model_out, target=target
                            )
                        else:
                            loss += self.__loss_compute_LSTMAE_UL(
                                model_output=model_out, target=target
                            )
                    if mode in (InTrainingMode.TRAIN.value,):

                        self.update_model(loss, lr)
                    loss_sum += loss.item()
                else:
                    weights_for_loss = [1 / len(self.modaility_set)] * len(
                        self.modaility_set
                    )
                    # Here, data is (modality1, modality2, .... , modalityN)
                    loss = None
                    self.alternavtie_alignment = 0
                    for local_idx_ori, ori in enumerate(self.modaility_set):
                        for local_idx_des, des in enumerate(self.modaility_set):
                            if len(selected_pairs) != 0 and f"{ori}-{des}" not in selected_pairs:
                                continue
                            des_local_idx = self.modaility_set.index(des)
                            model_out = [self.__forward_compute_LSTMAE_UL_AB(data, ori, des)]
                            target = [data[des_local_idx]]

                            if loss is None or self.alternavtie_alignment == 1:
                                loss = self.__loss_compute_LSTMAE_UL(
                                    mode=mode,
                                    model_output=model_out,
                                    target=target,
                                )
                            else:
                                loss += self.__loss_compute_LSTMAE_UL(
                                    mode=mode,
                                    model_output=model_out,
                                    target=target,
                                )
                            if self.alternavtie_alignment == 1:
                                loss_sum += loss.item()
                                if mode in (InTrainingMode.TRAIN.value,):
                                    self.update_model(loss, lr)

                    if self.alternavtie_alignment == 0:
                        loss_sum += loss.item()
                        if mode in (InTrainingMode.TRAIN.value,):
                            self.update_model(loss, lr)
                    # assert False

            elif self.learning_type == LearningType.UNSUPERVISED_AB.value:
                ori, des = self.__get_ori_des(extra_info)
                des_local_idx = self.modaility_set.index(des)
                model_out = [self.__forward_compute_LSTMAE_UL_AB(data, ori, des)]
                target = [data[des_local_idx]]
                loss = self.__loss_compute_LSTMAE_UL(
                    model_output=model_out, target=target, mode=mode
                )
                if mode in (InTrainingMode.TRAIN.value,):
                    self.update_model(loss, lr)
                loss_sum += loss.item()
            elif any(
                [
                    self.learning_type == LearningType.SUPERVISED_WITH_ENCODER.value,
                    self.learning_type == LearningType.SUPERVISED_FOR_AUX.value,
                    self.learning_type == LearningType.SUPERVISED.value,
                ]
            ):
                model_out = None
                target = None
                if self.learning_type == LearningType.SUPERVISED_WITH_ENCODER.value:
                    model_out = self.__forward_compute_SL_LSTMAE_Encoded(data)
                    target = label.to(device)
                    # print(target.shape)
                    # assert False

                    # model_out, targets = self.__forward_compute_SL_LSTMAE_Encoded2(
                    #     data, label.to(device)
                    # )
                    # target = targets
                    # model_out = self.__forward_compute_SL_LSTMAE_Encoded3(data[0]).to(
                    #     device
                    # )
                    # target = label.to(device)
                elif self.learning_type == LearningType.SUPERVISED_FOR_AUX.value:
                    _, des = self.__get_ori_des(extra_info)
                    dis_threshold = extra_info['dis_threshold'] 
                    mean_probs, m_cnt = 0, 0
                    model_out, target = None, None
                    for ori in self.modaility_set:
                        if ori == des:
                            continue
                        model_out_ab, target_ab, mean_probs_ab = self.__forward_compute_SL4AUX(
                            data, ori, des, batch_idx
                        )
                        m_cnt += 1
                        if model_out is None:
                            model_out = model_out_ab
                            target = target_ab
                            mean_probs = mean_probs_ab
                        else:
                            model_out = torch.concatenate([model_out, model_out_ab], dim=0)
                            target = torch.concatenate([target, target_ab], dim=0)
                            mean_probs += mean_probs_ab
                    mean_probs = mean_probs / m_cnt
                    if batch_idx == 0:
                        print("mean_probs", mean_probs)
                    if mean_probs > dis_threshold:
                        break

                elif self.learning_type == LearningType.SUPERVISED.value:
                    model_out = self.model(data[self.modality_choice_in_SL].to(device))
                    target = label.to(device)

                pred = None

                if (
                    self.dataset_name == DatasetName.CMUMOSEI.value
                    and self.learning_type
                    in (
                        LearningType.SUPERVISED_WITH_ENCODER.value,
                        LearningType.SUPERVISED.value,
                    )
                ):
                    mae = model_out.flatten() - target.flatten()
                    pred_record = torch.concatenate(
                        [pred_record, model_out.flatten()], dim=0
                    )
                    target_record = torch.concatenate(
                        [target_record, target.flatten()], dim=0
                    )
                    mae = torch.sum(torch.abs(pred_record - target_record)) / len(
                        pred_record
                    )
                else:
                    pred = model_out.argmax(-1).to(device)
                    # print(target)
                    pred_record = torch.concatenate([pred_record, pred], dim=0)
                    target_record = torch.concatenate([target_record, target], dim=0)
                    acc_sum = float(torch.eq(pred_record, target_record).int().sum())
                    f1_score = f1(pred_record, target_record)
                # 
                loss = self.__loss_compute_SL(model_output=model_out, target=target, target_long = self.dataset_name != DatasetName.CMUMOSEI.value)
                if mode in (InTrainingMode.TRAIN.value,):
                    # print("target", target, loss)
                    # print("model_out", model_out)
                    self.update_model(loss, lr)
                loss_sum += loss.item()

            elif self.learning_type == LearningType.UNSUPERVISED_WITH_AUX.value:
                ori, _ = self.__get_ori_des(extra_info)
                model_out, target = None, None
                for des in self.comp_modality_set:
                    model_out_ab = self.__forward_compute_LSTMAE_UL_with_aux(data, ori, des)
                    target_ab = torch.ones((len(model_out_ab),), dtype=torch.int64).to(device)
                    if model_out is None:
                        model_out = model_out_ab
                        target = target_ab
                    else:
                        model_out = torch.concatenate([model_out, model_out_ab], dim=0)
                        target = torch.concatenate([target, target_ab], dim=0)

                # pred = torch.softmax(model_out, -1).argmax(-1).to(device)
                # pred_record = torch.concatenate([pred_record, pred], dim=0)
                # target_record = torch.concatenate([target_record, target], dim=0)
                # acc_sum = float(torch.eq(pred_record, target_record).int().sum())
                loss = self.__loss_compute_UL_with_AUX(
                    model_output=model_out, target=target
                )
                if mode == InTrainingMode.TRAIN.value:
                    self.__update_model_encoder_only(loss, lr)
                loss_sum += loss.item()

            if mode == InTrainingMode.TRAIN.value:
                if batch_idx % log_interval == 0:
                    log_info = "Client:{}; Global Epoch: {}; Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.owner,
                        global_epoch,
                        local_epoch,
                        num_samples_iter,
                        len(dataloader.dataset),
                        100.0 * num_samples_iter / len(self.train_dataloader.dataset),
                        loss_sum / num_samples_iter,
                    )

                    if self.learning_type in (
                        LearningType.SUPERVISED_WITH_ENCODER.value,
                        LearningType.SUPERVISED_FOR_AUX.value,
                        LearningType.SUPERVISED.value,
                        LearningType.UNSUPERVISED_WITH_AUX.value,
                    ):
                        if (
                            self.dataset_name == DatasetName.CMUMOSEI.value
                            and self.learning_type
                            in (
                                LearningType.SUPERVISED_WITH_ENCODER.value,
                                LearningType.SUPERVISED.value,
                            )
                        ):
                            mae_info = "\tMAE: {:.6f}".format(
                                mae,
                            )
                            log_info = log_info + mae_info
                        else:
                            acc_info = "\tAcc: {:.6f}".format(
                                acc_sum / max(len(pred_record), 1),
                            )
                            f1_info = "\tF1: {:.6f}".format(
                                f1_score,
                            )
                            log_info = log_info + acc_info + f1_info
                        if self.learning_type == LearningType.SUPERVISED_FOR_AUX.value:
                            log_info = (
                                "{}-{};".format(extra_info["a"], extra_info["b"])
                                + log_info
                            )
                    print(log_info)
                    # print(pred)
                    # print(model_out)
            if (
                batch_id_left >= 0
                and batch_id_right >= 0
                and batch_idx == batch_id_right - 1
            ):
                break

        if (
            self.fl_subtype == FLSubType.SCAFFOLD.value
            and mode == InTrainingMode.TRAIN.value
        ):
            with torch.no_grad():
                self.update_control_variate(lr, batch_idx + 1)

        average_loss = 0 if num_samples_iter == 0 else loss_sum / num_samples_iter
        average_acc = 0 if len(pred_record) == 0 else acc_sum / len(pred_record)

        # print(pred_record[:100])
        # print(target_record[:100])
        print("acc_sum", acc_sum, "sample len", len(pred_record))

        print(f"{mode} average Loss: {average_loss}")
        if self.learning_type in (
            LearningType.SUPERVISED_WITH_ENCODER.value,
            LearningType.SUPERVISED_FOR_AUX.value,
            LearningType.SUPERVISED.value,
        ):
            if self.dataset_name == DatasetName.CMUMOSEI.value:
                print(f"{mode} mae: {mae}")
            else:
                from collections import Counter

                print(Counter([int(i) for i in pred_record]))
                print(f"{mode} average acc: {average_acc} average f1:{f1_score}")

        return average_loss, average_acc

    def train_model(
        self,
        num_epoch,
        log_interval,
        global_epoch,
        learning_rate,
        extra_info,
        learning_rate_decay="",
        batch_start_id=-1,
        batch_end_id=-1,
        selected_pairs = []
    ):
        ori_learning_rate_decay = self.learning_rate_decay
        if learning_rate_decay != "":
            self.learning_rate_decay = learning_rate_decay
        if batch_start_id != -1 and batch_end_id != -1:
            num_epoch = 1
        losses = []
        acc = []
        train_epoch_loss, train_epoch_acc = self.train_model_with_data_batches(
            mode=InTrainingMode.EVALUATE_WITH_TRAINING_DATA.value,
            batch_id_right=-1,
            batch_id_left=-1,
            extra_info=extra_info,
        )
        losses.append(train_epoch_loss)
        acc.append(train_epoch_acc)

        best_model_ever = None
        if self.learning_rate_decay == LearningRateDecay.SMART_RECALL.value:
            best_model_ever = deepcopy(self.model)

        ori_lr = self.learning_rate
        if learning_rate > 0:
            ori_lr = learning_rate

        print("local ori-learning_rate:", ori_lr, "decay:", self.learning_rate_decay)
        training_time = 0
        for epoch in range(num_epoch):
            lr = ori_lr
            if self.learning_rate_decay == LearningRateDecay.COSINE.value:
                lr = cosine_decay_lr(ori_lr, epoch, num_epoch)
                # lr = ori_lr * (0.1 ** (epoch // 80))
            print("local learning_rate:", lr)
            start_time = time.time()
            self.train_model_with_data_batches(
                InTrainingMode.TRAIN.value,
                batch_start_id,
                batch_end_id,
                global_epoch,
                epoch,
                log_interval,
                lr,
                extra_info,
                selected_pairs
            )
            end_time = time.time()
            training_time += end_time - start_time
            train_epoch_loss, train_epoch_acc = self.train_model_with_data_batches(
                mode=InTrainingMode.EVALUATE_WITH_TRAINING_DATA.value,
                batch_id_left=-1,
                batch_id_right=-1,
                extra_info=extra_info,
            )
            losses.append(train_epoch_loss)
            acc.append(train_epoch_acc)
            self.train_model_with_data_batches(
                mode=InTrainingMode.EVALUATE.value,
                batch_id_left=-1,
                batch_id_right=-1,
                extra_info=extra_info,
            )

            if self.learning_rate_decay == LearningRateDecay.SMART_RECALL.value:
                if LearningRateDecay.SMART_RECALL.value not in extra_info:
                    window_size = 4
                    pending_stage = 10
                    decay_ratio = 0.1
                else:
                    window_size = extra_info[LearningRateDecay.SMART_RECALL.value][
                        "window_size"
                    ]
                    pending_stage = extra_info[LearningRateDecay.SMART_RECALL.value][
                        "pending_stage"
                    ]
                    decay_ratio = extra_info[LearningRateDecay.SMART_RECALL.value][
                        "decay_ratio"
                    ]

                if min(losses) == train_epoch_loss:
                    if best_model_ever is not None:
                        del best_model_ever
                    print("find the best model ever!")
                    best_model_ever = deepcopy(self.model)

                max_loss = (
                    max(losses[:-1])
                    if len(losses) - 1 < window_size
                    else max(losses[-window_size - 1 : -1])
                )
                if len(losses) > pending_stage and train_epoch_loss > max_loss:
                    self.model.load_state_dict(best_model_ever.state_dict())
                    ori_lr = ori_lr * decay_ratio
                    print("recall the best model!", ori_lr)
                    losses.pop()
                    acc.pop()
        self.learning_rate_decay = ori_learning_rate_decay


        return {"losses":losses, "acc":acc, "reconstruction_loss":None, "training_time": training_time}

    def eval_reconstruction_loss(self):
        #get reconstruction loss for all modality pairs
        reconstruction_loss_dict = {}
        
        for ori in self.modaility_set:
            for des in self.modaility_set:
                reconstruction_loss_dict[f"{ori}-{des}"] = 0

        num_samples_iter = 0
        start = time.time()
        for batch_idx, data_ in enumerate(self.train_dataloader, 0):
            num_samples_iter += len(data_[0])
            data, label = data_preprocess_transposed(
                data_, self.to_transpose, self.dataset_info.use_time_step_dim
            )
            # data = [self.z_norm(x) for x in data]
            
            for ori in self.modaility_set:
                for des in self.modaility_set:
                    des_local_idx = self.modaility_set.index(des)
                    model_out = [self.__forward_compute_LSTMAE_UL_AB(data, ori, des)]
                    target = [data[des_local_idx]]
                    loss = self.__loss_compute_LSTMAE_UL(
                        model_output=model_out, target=target
                    )
                    reconstruction_loss_dict[f"{ori}-{des}"] += loss.item()
        
        for ori in self.modaility_set:
            for des in self.modaility_set:
                reconstruction_loss_dict[f"{ori}-{des}"] /= num_samples_iter
        training_time =  time.time()-start

        return {"reconstruction_loss": reconstruction_loss_dict, "sample_num": num_samples_iter, "training_time":training_time}
        


    def test_model(self, extra_info={}):
        return self.train_model_with_data_batches(
            mode=InTrainingMode.TEST.value,
            batch_id_left=-1,
            batch_id_right=-1,
            extra_info=extra_info,
        )
