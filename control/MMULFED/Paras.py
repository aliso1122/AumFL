from control.Enums import ModelType, DatasetName, LearningRateDecay, DatasetInfo
from typing import List


class AuxModelConfigs:
    def __init__(
        self, dataset_name: str, global_modality_set: tuple, AC_local_epoch: int = 30
    ):
        self.dataset_info = DatasetInfo(dataset_name)
        self.aux_model_configs: List[AuxModelConfig] = []
        for m in global_modality_set:
            self.aux_model_configs.append(
                AuxModelConfig(dataset_name, self.dataset_info, m, AC_local_epoch)
            )


class AuxModelConfig:
    def __init__(
        self,
        dataset_name: str,
        dataset_info: DatasetInfo,
        modality: int = 0,
        AC_local_epoch: int = 30,
    ):
        self.dataset_name = dataset_name
        self.modality = modality
        if dataset_name == DatasetName.CMUMOSEI.value:
            self.model_type = ModelType.LSTMDIS.value
            self.input_size = dataset_info.dims[modality]
            self.hidden_size = 256
            self.local_epoch_number = AC_local_epoch
            self.global_epoch_number = 1
            self.learning_rate = 0.1
            self.learning_rate_decay = LearningRateDecay.COSINE.value
        if dataset_name == DatasetName.URFALL.value:
            self.model_type = ModelType.LSTMDIS.value
            self.input_size = dataset_info.dims[modality]
            self.hidden_size = 64
            self.local_epoch_number = AC_local_epoch
            self.global_epoch_number = 1
            self.learning_rate = 0.1
            self.learning_rate_decay = LearningRateDecay.COSINE.value

        elif dataset_name in (DatasetName.HAR.value):
            self.model_type = ModelType.LSTMDIS.value
            self.input_size = 3
            self.hidden_size = 256
            # self.learning_rate_decay = LearningRateDecay.SMART_RECALL.value
            self.learning_rate_decay = LearningRateDecay.COSINE.value
            self.learning_rate = 0.1
            self.local_epoch_number = AC_local_epoch
            self.global_epoch_number = 1
            # if modality == 0:
            #     self.local_epoch_number = 10
            #     self.global_epoch_number = 1
            #     self.learning_rate = 0.5
            #     self.learning_rate_decay = LearningRateDecay.COSINE.value
            # elif modality == 1:
            #     self.learning_rate = 0.1
            #     self.learning_rate_decay = LearningRateDecay.COSINE.value
            # elif modality == 2:
            #     self.local_epoch_number = 10
            #     self.global_epoch_number = 1
            #     self.learning_rate = 0.5
            #     self.learning_rate_decay = LearningRateDecay.SMART_RECALL.value
            #     self.local_epoch_number2 = 1
            #     self.global_epoch_number2 = 46
            #     self.learning_rate2 = 0.00002
            #     self.learning_rate_decay2 = LearningRateDecay.COSINE.value
