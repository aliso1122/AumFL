from enum import Enum


class DataAnalysisDimensions(Enum):
    Model_Name = "model_name"
    LOG_TYPE = "log_type"
    ClIENT_ID = "client_id"
    GLOBAL_EPOCH = "global_epoch"
    EXTRA = "extra"


class DataAnalysisIndicators(Enum):
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    TRAIN_ACC = "train_acc"
    VAL_ACC = "val_acc"


class SpeiclaClientType(Enum):
    GLOBAL_CLIENT = "global_client"


class FLFramework(Enum):
    FEDAVG = "fedavg"
    MMULFED = "mmulfed"
    MMULFEDAA = "mmulfedaa"


class FLSubType(Enum):
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDUM = "fedum"


class LearningType(Enum):
    UNSUPERVISED = "unsupervised"
    UNSUPERVISED_AB = "unsupervised_ab"
    SUPERVISED = "supervised"
    SUPERVISED_WITH_ENCODER = "supervised_with_encoder"
    SUPERVISED_FOR_AUX = "supervised_for_aux"
    UNSUPERVISED_WITH_AUX = "unsupervised_with_aux"


class LearningRateDecay(Enum):
    NONE = "none"
    COSINE = "cosine"
    STAIR = "stair"
    SMART_RECALL = "smart_recall"


class ModelLoading(Enum):
    LOADNOTHING = 0
    LOADMODEL = 1


class DatasetName(Enum):
    HAR = "HAR"
    UCF101 = "UCF101"
    URFALL = "URFALL"
    CMUMOSEI = "CMUMOSEI"
    QILIN = "QILIN"


class DatasetInfo:
    def __init__(self, dataset_name: str, m_set: list = None):
        self.dataset_name = dataset_name
        self.batch_dimension_switched = 0

        if dataset_name in (
            DatasetName.HAR.value,
            DatasetName.UCF101.value,
            DatasetName.URFALL.value,
            DatasetName.CMUMOSEI.value,
        ):
            self.batch_dimension_switched = 1

        if dataset_name == DatasetName.HAR.value:
            self.use_time_step_dim = 0
            self.dims = [3, 3, 3]
            self.data_range_type = ["n1_to_p1", "n1_to_p1", "n1_to_p1"]
            if m_set is not None:
                self.input_sizes = [self.dims[m] for m in m_set]
                self.use_act_list = [
                    self.data_range_type[m] == "n1_to_p1" for m in m_set
                ]
        if dataset_name == DatasetName.CMUMOSEI.value:
            self.use_time_step_dim = 0
            self.dims = [74, 713, 300]
            self.data_range_type = ["normal", "normal", "normal"]
            if m_set is not None:
                self.input_sizes = [self.dims[m] for m in m_set]
                self.use_act_list = [
                    self.data_range_type[m] == "n1_to_p1" for m in m_set
                ]
        if dataset_name == DatasetName.UCF101.value:
            self.dims = [2048, 160]
            self.use_time_step_dim = 0
            self.data_range_type = ["normal", "n1_to_p1"]
            if m_set is not None:
                self.input_sizes = [self.dims[m] for m in m_set]
                self.use_act_list = [
                    self.data_range_type[m] == "n1_to_p1" for m in m_set
                ]
        if dataset_name == DatasetName.URFALL.value:
            self.dims = [3, 512, 8]
            self.use_time_step_dim = 1
            self.data_range_type = ["n1_to_p1", "normal", "n1_to_p1"]
            if m_set is not None:
                self.input_sizes = [self.dims[m] for m in m_set]
                self.use_act_list = [
                    self.data_range_type[m] == "n1_to_p1" for m in m_set
                ]


class DataPreprocessMethod(Enum):
    NONE = "none"


class ModelType(Enum):
    LSTMAE = "LSTMAE"
    MLP = "MLP"
    MLP1 = "MLP1"
    LSTMDIS = "LSTMDIS"
    LSTMCLASSIFY = "LSTMCLASSIFY"


class PurposeType(Enum):
    TRAIN = "train"
    PLOT = "plot"
    TEST = "test"
    GENERATE_DATA = "generate_data"


class DataSplitType(Enum):
    TYPE1 = "type1"

    TYPE3 = "type3"
    TYPE4 = "type4"
    TYPE5 = "type5"
    TYPE6 = "type6"
    SINGLE01 = "single01"
    SINGLE02 = "single02"
    SINGLE012 = "single012"
    HOMO01 = "homo01"
    HOMO02 = "homo02"
    HOMO012 = "homo012"
    HETERO02 = "hetero02"
    HETERO02_b = "hetero02_b"
    HETERO01 = "hetero01"
    HETERO012 = "hetero012"
    HETERO012_b = "hetero012_b"
    HUM02 = "hum02"
    HUM012 = "hum012"

    HETERO02_SINGLE = "hetero02_single"



def generate_client_id(
    modality_composition_type: int, number_under_modality_composition_type: int
):
    return f"{modality_composition_type}_{number_under_modality_composition_type}_"


def parse_client_id(client_id: str):
    a = client_id.split("_")
    return int(a[0]), int(a[1])


class ParametersForDataSplitType:
    def __init__(self, data_split_type: str, N_train: int = 0, N_test: int = 0):

        self.M_sets = None
        # D_len is the number of input pieces in a first splitted subdataset
        self.D_len_train = None
        self.D_len_test = None
        self.client_nums_for_Dk = None
        self.client_modality_choice_idx_in_SL = None

        if data_split_type == DataSplitType.HOMO012.value:
            self.M_sets = [(0, 1, 2)]
            self.D_len_train = [N_train]
            self.D_len_test = [N_test]
            self.client_nums_for_Dk = [10]
            self.client_modality_choice_idx_in_SL = [0]
        elif data_split_type == DataSplitType.HETERO012.value:
            self.M_sets = [
                (0, 1, 2),
                (0, 2),
                (1, 2),
                (0,),
                (1,),
            ]
            self.client_nums_for_Dk = [4, 3, 3, 3, 3]
            self.client_modality_choice_idx_in_SL = [0, 0, 0, 0, 0]
            self.D_len_train = [
                int(0.4 * N_train),
                int(0.3 * N_train),
                int(0.3 * N_train),
                int(0.3 * N_train),
                int(0.3 * N_train),
            ]
            self.D_len_test = [
                int(0.4 * N_test),
                int(0.3 * N_test),
                int(0.3 * N_test),
                int(0.3 * N_test),
                int(0.3 * N_test),
            ]
        elif data_split_type == DataSplitType.HETERO012_b.value:
            self.M_sets = [
                (0, 1),
                (0, 2),
                (1, 2),
                (0,),
                (1,),
                (2,),
            ]
            self.client_nums_for_Dk = [2, 2, 2, 6, 6, 6]
            self.client_modality_choice_idx_in_SL = [0, 0, 0, 0, 0, 0]
            self.D_len_train = [
                int(0.2 * N_train),
                int(0.2 * N_train),
                int(0.2 * N_train),
                int(0.6 * N_train),
                int(0.6 * N_train),
                int(0.6 * N_train),
            ]
            self.D_len_test = [
                int(0.2 * N_test),
                int(0.2 * N_test),
                int(0.2 * N_test),
                int(0.6 * N_test),
                int(0.6 * N_test),
                int(0.6 * N_test),
            ]
        elif data_split_type == DataSplitType.TYPE3.value:
            self.M_sets = [(0, 2), (0,), (2,)]
            self.client_nums_for_Dk = [1, 1, 1]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(0.5 * N_train),
                int(0.5 * N_train),
                int(0.5 * N_train),
            ]
            self.D_len_test = [
                int(0.5 * N_test),
                int(0.5 * N_test),
                int(0.5 * N_test),
            ]
        elif data_split_type == DataSplitType.TYPE4.value:
            self.M_sets = [(0, 2)]
            self.client_nums_for_Dk = [3]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
            ]
        elif data_split_type == DataSplitType.TYPE5.value:
            self.M_sets = [(0, 1)]
            self.client_nums_for_Dk = [1]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
            ]
        elif data_split_type == DataSplitType.SINGLE02.value:
            self.M_sets = [(0, 2)]
            self.client_nums_for_Dk = [1]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
            ]
        elif data_split_type == DataSplitType.HOMO02.value:
            self.M_sets = [(0, 2)]
            self.client_nums_for_Dk = [10]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [N_train]
            self.D_len_test = [N_test]
        elif data_split_type == DataSplitType.HOMO01.value:
            self.M_sets = [(0, 1)]
            self.client_nums_for_Dk = [10]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [N_train]
            self.D_len_test = [N_test]
        elif data_split_type == DataSplitType.HETERO02.value:
            self.M_sets = [(0, 2), (0,), (2,)]
            self.client_nums_for_Dk = [5, 5, 5]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(0.5 * N_train),
                int(0.5 * N_train),
                int(0.5 * N_train),
            ]
            self.D_len_test = [
                int(0.5 * N_test),
                int(0.5 * N_test),
                int(0.5 * N_test),
            ]
        elif data_split_type == DataSplitType.HETERO02_SINGLE.value:
            self.M_sets = [(0, 2), (0,), (2,)]
            self.client_nums_for_Dk = [1, 1,1]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(0.5 * N_train),
                int(0.5 * N_train),
                int(0.5 * N_train),
            ]
            self.D_len_test = [
                int(0.5 * N_test),
                int(0.5 * N_test),
                int(0.5 * N_test),
            ]
        
        elif data_split_type == DataSplitType.HUM02.value:
            self.M_sets = [(0,), (2,)]
            self.client_nums_for_Dk = [1, 1]
            self.client_modality_choice_idx_in_SL = [0, 0]
            self.D_len_train = [
                int(N_train),
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
                int(N_test),
            ]
        elif data_split_type == DataSplitType.HUM012.value:
            self.M_sets = [(0,), (1,), (2,)]
            self.client_nums_for_Dk = [1, 1, 1]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(N_train),
                int(N_train),
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
                int(N_test),
                int(N_test),
            ]
        elif data_split_type == DataSplitType.HETERO02_b.value:
            self.M_sets = [(0, 2), (0,), (2,)]
            self.client_nums_for_Dk = [1, 9, 9]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(0.1 * N_train),
                int(0.9 * N_train),
                int(0.9 * N_train),
            ]
            self.D_len_test = [
                int(0.1 * N_test),
                int(0.9 * N_test),
                int(0.9 * N_test),
            ]
        elif data_split_type == DataSplitType.HETERO01.value:
            self.M_sets = [(0, 1), (0,), (1,)]
            self.client_nums_for_Dk = [5, 5, 5]
            self.client_modality_choice_idx_in_SL = [0, 0, 0]
            self.D_len_train = [
                int(0.5 * N_train),
                int(0.5 * N_train),
                int(0.5 * N_train),
            ]
            self.D_len_test = [
                int(0.5 * N_test),
                int(0.5 * N_test),
                int(0.5 * N_test),
            ]
        elif data_split_type == DataSplitType.SINGLE01.value:
            self.M_sets = [(0, 1)]
            self.client_nums_for_Dk = [1]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
            ]
        elif data_split_type == DataSplitType.SINGLE012.value:
            self.M_sets = [(0, 1, 2)]
            self.client_nums_for_Dk = [1]
            self.client_modality_choice_idx_in_SL = [0]
            self.D_len_train = [
                int(N_train),
            ]
            self.D_len_test = [
                int(N_test),
            ]
        else:
            assert False
        assert len(self.M_sets) == len(self.client_nums_for_Dk) and len(
            self.M_sets
        ) == len(self.D_len_train)

        self.global_m_set = []
        for m_set in self.M_sets:
            for m in m_set:
                if m not in self.global_m_set:
                    self.global_m_set.append(m)
        self.global_m_set = tuple(self.global_m_set)

        self.modality_2_clients = {}
        for m in self.global_m_set:
            self.modality_2_clients[m] = []
            for k, M_set in enumerate(self.M_sets):
                if m in M_set:
                    for p in range(self.client_nums_for_Dk[k]):
                        self.modality_2_clients[m].append(generate_client_id(k, p))


class InTrainingMode(Enum):
    """
    Only used in Manager!!!
    """

    TRAIN = "train"
    EVALUATE = "evaluate"
    EVALUATE_WITH_TRAINING_DATA = "evaluate_with_training_data"
    TEST = "test"


class TrainingMode4LSTMAE(Enum):
    UNCONDITIONED = "unconditioned"
    TEACHER_FORCING = "teacher_forcing"
    SPECIAL_UNCONDITIONED = "special_unconditioned"
    MIXED_TEACHER_FORCING = "mixed_teacher_forcing"
    RECURSIVE = "recursive"
