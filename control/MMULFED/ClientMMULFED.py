from control.Client import Client
from control.Manager import HyperInfo, Manager
from copy import deepcopy
from control.Enums import LearningType
from models.AEwithAux import AEwithAux
from typing import List
from control.MMULFED.Paras import AuxModelConfig

class ClientMMULFED(Client):
    def __init__(
        self,
        hyper_info: HyperInfo,
        client_id: str,
        aux_model_hyperinfo_list: list,
    ):
        super(ClientMMULFED, self).__init__(hyper_info, client_id)
        self.aux_managers: List[Manager] = []
        for aux_model_hyperinfo in aux_model_hyperinfo_list:
            aux_model_hyperinfo.train_dataloader = self.manager.train_dataloader
            aux_model_hyperinfo.test_dataloader = self.manager.test_dataloader
            aux_model_hyperinfo.val_dataloader = self.manager.val_dataloader
            aux_model_hyperinfo.modaility_set = self.manager.modaility_set
            aux_model_hyperinfo.model.set_name(
                f"{client_id};{aux_model_hyperinfo.model_type}"
            )
            aux_manager = Manager(aux_model_hyperinfo, client_id)
            aux_manager.learning_type = LearningType.SUPERVISED_FOR_AUX.value
            aux_manager.generator = self.manager.model
            self.aux_managers.append(aux_manager)

        self.aux_models = [manager.model for manager in self.aux_managers]
        if (
            len(self.manager.modaility_set) < len(self.manager.global_modaility_set)
            or True
        ):
            self.manager.ae_aux_model = AEwithAux(
                self.manager.model.encoder_list,
                self.manager.model.decoder_list,
                self.aux_models,
                f"{client_id};ae-aux",
            )

    def train_with_aux(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters_dict: dict,
        global_epoch: int,
        local_epoch: int,
        log_interval: int,
        learning_rate: float = 0,
        batch_start_id: int = -1,
        batch_end_id: int = -1,
    ):
        self.manager.model.load_state_dict(global_model_parameters)
        for des in self.manager.comp_modality_set:
            des_global_idx = self.manager.global_modaility_set.index(des)
            aux_model = self.aux_models[des_global_idx]
            aux_model.load_state_dict(global_aux_model_parameters_dict[des].state_dict())

        self.manager.reset_environment(LearningType.UNSUPERVISED_WITH_AUX.value)
        for ori in self.manager.modaility_set:
            print("start-train-with-aux", self.client_id, "........")
            res = self.manager.train_model(
                num_epoch=local_epoch,
                log_interval=log_interval,
                global_epoch=global_epoch,
                learning_rate=learning_rate,
                extra_info={"a":ori, "b":-1},
                batch_start_id=batch_start_id,
                batch_end_id=batch_end_id,
            )
        train_losses, train_acc, training_time = res["losses"], res["acc"], res["training_time"]
        val_losses, val_acc = self.manager.test_model(extra_info={"a":ori, "b":-1})
        self.manager.reset_environment(LearningType.UNSUPERVISED.value)
        return self.manager.model.encoder_list, train_losses, train_acc, val_losses, val_acc, training_time

    def aux_train(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters_dict: dict,
        global_epoch: int,
        local_epoch: int,
        log_interval: int,
        aux_model_config_dict:dict,
        dis_threshold:float,
        batch_start_id: int = -1,
        batch_end_id: int = -1,
    ):
        returned_model = {}
        returned_info = {}
        for des in self.manager.modaility_set:
           
            des_idx = self.manager.global_modaility_set.index(des)
            aux_model_config: AuxModelConfig = aux_model_config_dict[des]

            learning_rate = aux_model_config.learning_rate
            learning_rate_decay = aux_model_config.learning_rate_decay
            
            aux_manager: Manager = self.aux_managers[des_idx]
            aux_manager.model.load_state_dict(global_aux_model_parameters_dict[des].state_dict())
            aux_manager.generator.load_state_dict(global_model_parameters)
            aux_manager.generator.requires_grad_(False)
            # ALL TRAININGS ARE DONE HERE! And losses is a list recording all training losses of all epochs; so is acc
            print("start-aux-train", self.client_id, "........ modality", des)
            res = aux_manager.train_model(
                num_epoch=local_epoch,
                log_interval=log_interval,
                global_epoch=global_epoch,
                learning_rate=learning_rate,
                extra_info={"a":-1, "b": des, "dis_threshold":dis_threshold},
                learning_rate_decay=learning_rate_decay,
                batch_start_id=batch_start_id,
                batch_end_id=batch_end_id,
            )
            train_losses, train_acc, training_time = res["losses"], res["acc"], res["training_time"]
            val_losses, val_acc = aux_manager.test_model({"a":-1, "b": des, "dis_threshold":dis_threshold})
            returned_model[des] = {
                "aux_model": aux_manager.model,
            }
            returned_info[des] = {
                "val_losses": val_losses,
                "val_acc": val_acc,
            }
        return returned_model, returned_info, training_time

    def aux_test(
        self,
        global_model_parameters: dict,
        global_aux_model_parameters: dict,
        extra_info: dict = {},
    ):
        print("start-aux-test", self.client_id, "........")
        des = extra_info["b"]
        des_idx = self.manager.global_modaility_set.index(des)
        aux_manager: Manager = self.aux_managers[des_idx]
        aux_manager.model.load_state_dict(global_aux_model_parameters)
        aux_manager.generator.load_state_dict(global_model_parameters)
        aux_manager.generator.requires_grad_(False)
        val_losses, val_acc = aux_manager.test_model(extra_info)
        return (
            val_losses,
            val_acc,
        )

    def rename(self, name: str):
        self.client_id = name
        self.manager.owner = name
        self.aux_manager.owner = name

    # def test(self):
    #     print("start test", self.client_id, "........")
    #     val_losses, val_acc = self.manager.test_model()
    #     return (
    #         val_losses,
    #         val_acc,
    #     )
