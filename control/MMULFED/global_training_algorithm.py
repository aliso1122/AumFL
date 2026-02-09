from torch import nn
from control.Enums import (
    ParametersForDataSplitType,
    FLFramework,
    DataAnalysisDimensions,
    InTrainingMode,
    LearningRateDecay,
    DataAnalysisIndicators,
    LearningType,
    DatasetInfo,
)
from control.MMULFED.Paras import AuxModelConfig
from typing import List
from control.globalside_toolkit import (
    make_client_group,
    sample_clients_normal,
    collect_encoders_decoders,
    aggregate_model_paras_in_list,
    parse_client_id,
    deserialize_model,
    serialize_model,
    aggregate_collected_encoders_decoders,
)
from control.MMULFED.toolkit import sample_clients_by_modalities
from control.Client import Client
from control.MMULFED.ClientMMULFED import ClientMMULFED
from control.global_training_algorithm import FLAlgorithmInfo, global_test_func
from tools.utils import get_log_info, cosine_decay_lr
from tools.plot_tools import plot_orig_vs_reconstructed


def unsupervised_global_training_with_discriminator(
    C_a_only: list, fl: FLAlgorithmInfo, aux_model: nn.Module, extra_info: dict
):
    """
    Unsupervised training with the discriminator DIS-b for
    the clients only having modality-a
    """
    epoch: int = extra_info["epoch"]
    a: int = extra_info["a"]
    b: int = extra_info["b"]
    a_global_idx: int = extra_info["a_global_idx"]
    b_global_idx: int = extra_info["b_global_idx"]
    encoders_a_2 = []
    for client_id in C_a_only:
        client: ClientMMULFED = fl.clients[client_id]
        (
            client_model,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ) = client.train_with_aux(
            global_model_parameters=fl.model.state_dict(),
            global_aux_model_parameters=aux_model.state_dict(),
            global_epoch=epoch,
            local_epoch=fl.AC_local_epoch,
            log_interval=fl.log_interval,
            extra_info=extra_info,
            learning_rate=fl.learning_rate,
        )
        fl.log_func(
            get_log_info(
                log_type=InTrainingMode.TRAIN.value,
                global_epoch=epoch,
                client_id=client_id,
                in_training_metrics=(
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ),
                metrics_name=(
                    DataAnalysisIndicators.TRAIN_LOSS.value,
                    DataAnalysisIndicators.TRAIN_ACC.value,
                    DataAnalysisIndicators.VAL_LOSS.value,
                    DataAnalysisIndicators.VAL_ACC.value,
                ),
                model_name=fl.model_type + "-aux",
                extra=f"{a}-{b}",
            )
        )
        # decoder_a_before = serialize_model(fl.model.decoder_list[a_global_idx])
        # decoder_a_after = serialize_model(client_model.decoder_list[a_global_idx])
        # decoder_b_before = serialize_model(fl.model.decoder_list[b_global_idx])
        # decoder_b_after = serialize_model(client_model.decoder_list[b_global_idx])
        # print(decoder_a_before - decoder_a_after)
        # print(decoder_b_before - decoder_b_after)
        # assert False
        encoders_a_2.append(serialize_model(client_model.encoder_list[a_global_idx]))
    aggregated_encoders_a_2 = aggregate_model_paras_in_list(encoders_a_2)
    deserialize_model(fl.model.encoder_list[a_global_idx], aggregated_encoders_a_2)


def supervised_global_training_for_discriminator(
    client_set: list,
    fl: FLAlgorithmInfo,
    extra_info: dict,
    non_iid_train: bool,
    dis_traininig_manage_info: dict,
):
    """
    Supervised training for the discriminator DIS-b and the
    tranining is alternating in the clients
    """
    epoch: int = extra_info["epoch"]
    a: int = extra_info["a"]
    b: int = extra_info["b"]
    b_global_idx: int = extra_info["b_global_idx"]
    aux_model = fl.aux_models[b_global_idx]
    aux_model_config = fl.aux_model_config_list[b_global_idx]
    batch_start_id = -1
    batch_num = 0
    ori_lr = aux_model_config.learning_rate
    decay = aux_model_config.learning_rate_decay
    aux_global_epoch_number = aux_model_config.global_epoch_number
    aux_local_epoch_number = aux_model_config.local_epoch_number
    if non_iid_train:
        batch_start_id = 0
        batch_num = 1
        ori_lr = aux_model_config.learning_rate2
        decay = aux_model_config.learning_rate_decay2
        aux_global_epoch_number = aux_model_config.global_epoch_number2
        aux_local_epoch_number = aux_model_config.local_epoch_number2

    for aux_epoch in range(aux_global_epoch_number):
        if decay == LearningRateDecay.COSINE.value:
            lr = cosine_decay_lr(
                ori_lr,
                aux_epoch,
                aux_global_epoch_number,
            )
        elif decay == LearningRateDecay.SMART_RECALL.value:
            lr = ori_lr
            extra_info[LearningRateDecay.SMART_RECALL.value] = {
                "window_size": 4,
                "pending_stage": (0 if dis_traininig_manage_info[b]["trained"] else 10),
                "decay_ratio": 0.3,
            }
        else:
            lr = ori_lr
        # if aux_epoch >= 0:
        #     lr = 0.00002
        aux_model_cache = []
        for client_id in client_set:
            client: ClientMMULFED = fl.clients[client_id]
            (
                client_aux_model,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            ) = client.aux_train(
                global_model_parameters=fl.model.state_dict(),
                global_aux_model_parameters=aux_model.state_dict(),
                global_epoch=aux_epoch,
                local_epoch=aux_local_epoch_number,
                log_interval=fl.log_interval,
                learning_rate=lr,
                extra_info=extra_info,
                learning_rate_decay=decay,
                batch_start_id=batch_start_id,
                batch_end_id=batch_start_id + batch_num,
            )
            if non_iid_train:
                batch_start_id += batch_num
            fl.log_func(
                get_log_info(
                    log_type=InTrainingMode.TRAIN.value,
                    global_epoch=epoch,
                    client_id=client_id,
                    in_training_metrics=(
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                    ),
                    metrics_name=(
                        DataAnalysisIndicators.TRAIN_LOSS.value,
                        DataAnalysisIndicators.TRAIN_ACC.value,
                        DataAnalysisIndicators.VAL_LOSS.value,
                        DataAnalysisIndicators.VAL_ACC.value,
                    ),
                    model_name=aux_model_config.model_type + str(b),
                    extra=f"{a}-{b}",
                )
            )
            aux_model_cache.append(serialize_model(client_aux_model))
        aggregated_aux_model = aggregate_model_paras_in_list(aux_model_cache)
        deserialize_model(aux_model, aggregated_aux_model)
        dis_traininig_manage_info[b]["trained"] = True
        # val_acc, val_loss = fl.global_client.aux_test(
        #     global_model_parameters=fl.model.state_dict(),
        #     global_aux_model_parameters=aux_model.state_dict(),
        #     extra_info=extra_info,
        # )
        # fl.log_func(
        #     get_log_info(
        #         log_type=InTrainingMode.TRAIN.value,
        #         global_epoch=aux_epoch,
        #         client_id=fl.global_client.client_id,
        #         in_training_metrics=(
        #             [0],
        #             [0],
        #             val_loss,
        #             val_acc,
        #         ),
        #         metrics_name=(
        #             DataAnalysisIndicators.TRAIN_LOSS.value,
        #             DataAnalysisIndicators.TRAIN_ACC.value,
        #             DataAnalysisIndicators.VAL_LOSS.value,
        #             DataAnalysisIndicators.VAL_ACC.value,
        #         ),
        #         model_name=aux_model_config.model_type + str(b),
        #         extra=f"{a}-{b}",
        #     )
        # )


def normal_unsupervised_global_training(
    C_ab: list, fl: FLAlgorithmInfo, extra_info: dict
):
    """
    Normal unsupervised training for the modality-complete clients
    """

    epoch = extra_info["epoch"]
    a = extra_info["a"]
    b = extra_info["b"]
    a_global_idx = extra_info["a_global_idx"]
    b_global_idx = extra_info["b_global_idx"]
    epoch = extra_info["epoch"]
    encoders_a = []
    decoders_b = []
    lr = fl.learning_rate * 5 if a == b else fl.learning_rate
    for client_id in C_ab:
        client: ClientMMULFED = fl.clients[client_id]
        res = client.train(
            global_model_parameters=fl.model.state_dict(),
            global_epoch=epoch,
            local_epoch=fl.local_epoch_number,
            log_interval=fl.log_interval,
            learning_type=LearningType.UNSUPERVISED_AB.value,
            extra_info=extra_info,
            learning_rate=lr
        )

        client_model,train_loss,train_acc,val_loss,val_acc = res['returned_model'],res['train_losses'],res['train_acc'],res['val_losses'],res['val_acc']

        fl.log_func(
            get_log_info(
                log_type=InTrainingMode.TRAIN.value,
                global_epoch=epoch,
                client_id=client_id,
                in_training_metrics=(
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ),
                metrics_name=(
                    DataAnalysisIndicators.TRAIN_LOSS.value,
                    DataAnalysisIndicators.TRAIN_ACC.value,
                    DataAnalysisIndicators.VAL_LOSS.value,
                    DataAnalysisIndicators.VAL_ACC.value,
                ),
                model_name=fl.model_type,
                extra=f"{a}-{b}",
            )
        )
        encoders_a.append(serialize_model(client_model.encoder_list[a_global_idx]))
        decoders_b.append(serialize_model(client_model.decoder_list[b_global_idx]))
    aggregated_encoders_a = aggregate_model_paras_in_list(encoders_a)
    aggregated_decoders_b = aggregate_model_paras_in_list(decoders_b)
    deserialize_model(fl.model.encoder_list[a_global_idx], aggregated_encoders_a)
    deserialize_model(fl.model.decoder_list[b_global_idx], aggregated_decoders_b)
    # val_loss, val_acc = fl.global_client.test(
    #     global_model_parameters=fl.model.state_dict(),
    # )
    # fl.log_func(
    #     get_log_info(
    #         log_type=InTrainingMode.TRAIN.value,
    #         global_epoch=epoch,
    #         client_id=fl.global_client.client_id,
    #         in_training_metrics=(
    #             [0],
    #             [0],
    #             val_loss,
    #             val_acc,
    #         ),
    #         metrics_name=(
    #             DataAnalysisIndicators.TRAIN_LOSS.value,
    #             DataAnalysisIndicators.TRAIN_ACC.value,
    #             DataAnalysisIndicators.VAL_LOSS.value,
    #             DataAnalysisIndicators.VAL_ACC.value,
    #         ),
    #         model_name=fl.model_type,
    #     )
    # )


def multimodal_unsupervised_learning_global_MMUL(fl: FLAlgorithmInfo):
    assert fl.algorithm_name == FLFramework.MMULFED.value
    global_m_set = fl.split_parameters.global_m_set
    dis_traininig_manage_info = {}
    for m in global_m_set:
        dis_traininig_manage_info[m] = {"trained": False}
    for epoch in range(fl.global_epoch_number):
        for a_global_idx, a in enumerate(global_m_set):
            for b_global_idx, b in enumerate(global_m_set):
                extra_info = {
                    "a": a,
                    "b": b,
                    "epoch": epoch,
                    "a_global_idx": a_global_idx,
                    "b_global_idx": b_global_idx,
                }
                # if a == b:
                #     continue

                C_ab, C_a_only, C_b_only = sample_clients_by_modalities(
                    a,
                    b,
                    fl.client_num,
                    max(fl.client_num // 2, 1),
                    max(fl.client_num // 2, 1),
                    fl.split_parameters,
                )

                fl.log_func(
                    get_log_info(
                        log_type="global",
                        global_epoch=epoch,
                        client_id=None,
                        in_training_metrics=(
                            None,
                            None,
                            None,
                            None,
                        ),
                        metrics_name=(
                            DataAnalysisIndicators.TRAIN_LOSS.value,
                            DataAnalysisIndicators.TRAIN_ACC.value,
                            DataAnalysisIndicators.VAL_LOSS.value,
                            DataAnalysisIndicators.VAL_ACC.value,
                        ),
                        model_name="",
                        extra="{}-{};{}-{}-{}".format(a, b, C_ab, C_a_only, C_b_only),
                    )
                )
                if len(C_ab) == 0:
                    continue
                normal_unsupervised_global_training(
                    C_ab=C_ab, fl=fl, extra_info=extra_info
                )
                if a == b:
                    # global_test_func(
                    #     global_client=fl.global_client,
                    #     global_model_paras=fl.model.state_dict(),
                    #     log_func=fl.log_func,
                    #     info_dict={"epoch": epoch, "model_name": fl.model_type},
                    # )
                    continue
                if fl.with_aux == 1:
                    supervised_global_training_for_discriminator(
                        client_set=[C_ab[0]],
                        fl=fl,
                        extra_info=extra_info,
                        non_iid_train=False,
                        dis_traininig_manage_info=dis_traininig_manage_info,
                    )
                    unsupervised_global_training_with_discriminator(
                        C_a_only=C_a_only,
                        fl=fl,
                        aux_model=fl.aux_models[b_global_idx],
                        extra_info=extra_info,
                    )

                # plot_orig_vs_reconstructed(
                #     local_modailities=fl.split_parameters.global_m_set,
                #     global_modailities=fl.split_parameters.global_m_set,
                #     model=fl.model,
                #     model_dir=fl.model_dir,
                #     test_iter=fl.global_client.get_test_loader(),
                #     dataset_info=DatasetInfo(dataset_name=fl.dataset_name),
                #     assigned_ori_modality=0,
                #     assigned_des_modality=2,
                #     save_path=f"{fl.model_dir}/log/{epoch}.png",
                #     num_to_plot=1,
                # )

        reconstruction_loss = fl.global_client.eval_reconstruction(fl.model.state_dict())["reconstruction_loss"]
        fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "RECONSTRUCTION", "reconstruction_loss": reconstruction_loss, "epoch": epoch})
