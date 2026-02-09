import torch
from control.Server import Server
from control.globalside_toolkit import (
    get_model,
    create_clients,
    get_local_model,
)
from control.global_training_algorithm import (
    multimodal_unsupervised_learning_global_default,
    multimodal_supervised_learning_global_default,
    FLAlgorithmInfo,
)

from control.MMULFED.global_training_algorithm import (
    multimodal_unsupervised_learning_global_MMUL,
)
from control.MMULFED.alignment_aware_global_training_algorithm import (
    multimodal_unsupervised_learning_global_MMULAA
)
from control.MMULFED.Paras import AuxModelConfigs
from control.Enums import (
    PurposeType,
    ModelLoading,
    DataAnalysisDimensions,
    LearningType,
    LearningRateDecay,
    FLFramework,
    InTrainingMode,
)
from control.Manager import HyperInfo
import os
from control.Client import Client
from tools.utils import get_log_info
from copy import deepcopy


class ServerMMULFED(Server):
    def __init__(self, args):
        super(ServerMMULFED, self).__init__(args)

        aux_model_configs = AuxModelConfigs(
            self.args.dataset_name, self.global_modality_set, self.args.AC_local_epoch
        )
        self.aux_model_configs_list = aux_model_configs.aux_model_configs
        self.aux_model_list = []
        self.__init_aux_model()

    def __init_aux_model(self):
        for i in range(len(self.global_modality_set)):
            dict_para = vars(self.aux_model_configs_list[i])
            dict_para["data_split_type"] = self.args.data_split_type
            self.aux_model_list.append(get_model(dict_para, "y"))

        # if (
        #     self.args.purpose == PurposeType.TRAIN.value
        #     and self.args.load_aux_model == ModelLoading.LOADMODEL.value
        # ):
        #     load_aux_model_path = f"{self.args.model_dir}/{self.args.load_aux_model_name}/{self.args.load_aux_model_name}.pt"
        #     self.aux_model.load_state_dict(torch.load(load_aux_model_path))
        #     print("successfully load model " + load_aux_model_path)

    def train(self):
        hyper_info = HyperInfo(
            learning_type=self.args.learning_type,
            modality_choice_idx_in_SL=None,
            model_type=self.args.model_type,
            dataset_name=self.args.dataset_name,
            data_split_type=self.args.data_split_type,
            modaility_set=None,
            global_modaility_set=self.global_modality_set,
            model=self.model,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=None,
            criterion=torch.nn.MSELoss(reduction="sum"),
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            encoder=None,
            encoder_info=None,
            data_preprocess_method=None,
        )

        aux_model_hyperinfo_list = [
            HyperInfo(
                learning_type=LearningType.SUPERVISED_FOR_AUX.value,
                modality_choice_idx_in_SL=None,
                model_type=self.aux_model_configs_list[i].model_type,
                dataset_name=self.args.dataset_name,
                data_split_type=None,
                modaility_set=None,
                global_modaility_set=self.global_modality_set,
                model=self.aux_model_list[i],
                train_dataloader=None,
                val_dataloader=None,
                test_dataloader=None,
                criterion=torch.nn.CrossEntropyLoss(),
                learning_rate=self.aux_model_configs_list[i].learning_rate,
                learning_rate_decay=self.aux_model_configs_list[i].learning_rate_decay,
                encoder=None,
                encoder_info=None,
                data_preprocess_method=None,
            )
            for i in range(len(self.global_modality_set))
        ]
        self.__init_clients(hyper_info, aux_model_hyperinfo_list)
        self.create_global_test_client(hyper_info, aux_model_hyperinfo_list)

        # aux_model_path = self.model_dir_with_id + "/" + "aux_model.pt"
        # self.aux_model.load_state_dict(torch.load(aux_model_path))
        if self.args.purpose == PurposeType.PLOT.value:
            return
        if self.args.purpose == PurposeType.TEST.value:
            for client_id in self.clients:
                client: Client = self.clients[client_id]
                loss, acc = client.test()
                self.log(
                    get_log_info(
                        log_type=InTrainingMode.TEST.value,
                        global_epoch=None,
                        client_id=client_id,
                        in_training_metrics=(loss, acc),
                        metrics_name=("loss", "acc"),
                    )
                )
                reconstruction_loss = client.eval_reconstruction()["reconstruction_loss"]
                self.log(
                    {DataAnalysisDimensions.LOG_TYPE.value: "RECONSTRUCTION_TEST", "reconstruction_loss": reconstruction_loss, "client_id": client_id}
                )
            return
        if self.args.load_aux_model == 1:
            self.__load_aux_models()
        fl_info_ul = FLAlgorithmInfo(
            algorithm_name=self.args.fl_framework,
            dataset_name=self.args.dataset_name,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            global_epoch_number=self.args.global_epoch_number,
            local_epoch_number=self.args.local_epoch_number,
            log_interval=self.args.log_interval,
            channel_num=self.args.channel_num,
            clients=self.clients,
            global_client=self.client_for_global_test,
            client_num=self.args.client_num_per_epoch,
            model=self.model,
            model_type=self.args.model_type,
            model_dir=self.model_level_dir,
            split_parameters=self.parameters,
            log_func=self.log,
            save_func=self.save_as_pickle,
            with_aux=self.args.with_aux,
            AC_local_epoch=self.args.AC_local_epoch,
            aux_models=self.aux_model_list,
            aux_model_config_list=self.aux_model_configs_list,
            aa_dropout_eps1=self.args.aa_dropout_eps1,
            aa_dropout_pvalue1=self.args.aa_dropout_pvalue1,
            aa_dropout_eps2=self.args.aa_dropout_eps2,
            aa_dropout_pvalue2=self.args.aa_dropout_pvalue2,
            aa_dropout_L=self.args.aa_dropout_L,
            aa_dropout_T_imp=self.args.aa_dropout_T_imp,
            aa_dropout_frozen_length=self.args.aa_dropout_frozen_length,
            aa_AC_gap = self.args.AC_gap,
            dis_threshold=self.args.dis_threshold
        )
        if self.args.fl_framework == FLFramework.MMULFED.value:
            multimodal_unsupervised_learning_global_MMUL(fl_info_ul)
        elif self.args.fl_framework == FLFramework.MMULFEDAA.value:
            multimodal_unsupervised_learning_global_MMULAA(fl_info_ul)
        self.__save_aux_models()

    def __init_clients(self, hyper_info: HyperInfo, aux_model_hyperinfo_list: list):
        self.clients = create_clients(
            global_M_set=self.global_modality_set,
            M_sets=self.parameters.M_sets,
            client_nums_for_Dk=self.parameters.client_nums_for_Dk,
            client_modality_choice_idx_in_SL=self.parameters.client_modality_choice_idx_in_SL,
            fl_framework=self.args.fl_framework,
            hyper_info=hyper_info,
            aux_model_hyperinfo_list=aux_model_hyperinfo_list,
        )

    def __save_aux_models(self):
        for i, modality in enumerate(self.global_modality_set, 0):
            aux_model = self.aux_model_list[i]
            torch.save(
                aux_model.state_dict(),
                os.path.join(
                    self.model_level_dir,
                    f"DIS{modality}.pt",
                ),
            )

    def __load_aux_models(self):
        for i, modality in enumerate(self.global_modality_set, 0):
            aux_model = self.aux_model_list[i]
            load_aux_model_path = f"{self.model_level_dir}/DIS{modality}.pt"
            aux_model.load_state_dict(torch.load(load_aux_model_path))
