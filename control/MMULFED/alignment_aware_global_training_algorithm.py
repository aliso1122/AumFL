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
from control.MMULFED.toolkit import trend_slope,sample_useful_clients
from control.Client import Client
from control.MMULFED.ClientMMULFED import ClientMMULFED
from control.global_training_algorithm import FLAlgorithmInfo, global_test_func, init_alignment_loss_records, collect_alignment_loss_records, get_one_alignment_loss_average, get_all_alignment_loss_average
from tools.utils import get_log_info, cosine_decay_lr
from tools.plot_tools import plot_orig_vs_reconstructed

import pandas as pd

def alignment_dropout_converged(alignment_loss_average, e, eps1, p_value_threshold1, eps2, p_value_threshold2, dropout_L_dict,):
    dropouts = {}
    # print(eps, p_value_threshold)
    for k in alignment_loss_average:
        loss_seq_length = len(alignment_loss_average[k])
        slope1, p_value1 = None, None
        con_drop = False
        if loss_seq_length < dropout_L_dict[k]:
            con_drop = False
        elif loss_seq_length >= dropout_L_dict[k]:
            slope1, p_value1 = trend_slope(alignment_loss_average[k], dropout_L_dict[k])
            con_drop = abs(slope1) < eps1 and p_value1 > p_value_threshold1
        
        slope2, p_value2 = None, None
        imp_drop = False
        if loss_seq_length < dropout_L_dict[k]:
            imp_drop = False
        elif loss_seq_length >= dropout_L_dict[k]:
            slope2, p_value2 = trend_slope(alignment_loss_average[k], loss_seq_length)
            imp_drop = slope2 > eps2 and p_value2 < p_value_threshold1

        dropouts[k] = { 
            "epoch":e,
            "con_drop": con_drop,
            "slope1": slope1,
            "p_value1": p_value1,
            "eps1": eps1,
            "p_value_threshold1": p_value_threshold1,
            "L": dropout_L_dict[k],
            "imp_drop": imp_drop,
            "slope2": slope2, 
            "p_value2": p_value2, 
            "eps2": eps2, 
            "p_value_threshold2": p_value_threshold1,
            }
        
    return dropouts

def unsupervised_global_training_with_discriminator(
    C_to_comp: list, fl: FLAlgorithmInfo, extra_info: dict
):
    epoch: int = extra_info["epoch"]

    aux_model_dict = {}
    for b in fl.split_parameters.global_m_set:
        b_global_idx = fl.split_parameters.global_m_set.index(b)
        aux_model_dict[b] = fl.aux_models[b_global_idx]

    encoders_dict = {m:[] for m in fl.split_parameters.global_m_set}
    for client_id in C_to_comp:
        client: ClientMMULFED = fl.clients[client_id]
        (
            client_encoders,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            training_time
        ) = client.train_with_aux(
            global_model_parameters=fl.model.state_dict(),
            global_aux_model_parameters_dict=aux_model_dict,
            global_epoch=epoch,
            local_epoch=fl.AC_local_epoch,
            log_interval=fl.log_interval,
            learning_rate=fl.learning_rate,
        )
        fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "TRAIN_WITH_AUX",  "epoch": epoch, "client_id": client_id,  "training_time": training_time})
        for a in client.get_modality_set():
            a_global_idx = fl.split_parameters.global_m_set.index(a)
            encoders_dict[a].append(serialize_model(client_encoders[a_global_idx]))

    for a in fl.split_parameters.global_m_set:
        if len(encoders_dict[a])==0:
            continue
        a_global_idx = fl.split_parameters.global_m_set.index(a)
        aggregated_encoders_a = aggregate_model_paras_in_list(encoders_dict[a])
        deserialize_model(fl.model.encoder_list[a_global_idx], aggregated_encoders_a)


def supervised_global_training_for_discriminator(
    client_set: list,
    fl: FLAlgorithmInfo,
    extra_info: dict,
    non_iid_train: bool=False,
):
    """
    Supervised training for the discriminator DIS-b
    """
    epoch: int = extra_info["epoch"]

    aux_model_dict = {}
    aux_model_config_dict = {}
    for b in fl.split_parameters.global_m_set:
        b_global_idx = fl.split_parameters.global_m_set.index(b)
        aux_model_dict[b] = fl.aux_models[b_global_idx]
        aux_model_config_dict[b] = fl.aux_model_config_list[b_global_idx]
    
    batch_start_id = -1
    batch_num = 0
    # ori_lr = aux_model_config.learning_rate
    # decay = aux_model_config.learning_rate_decay
    # aux_global_epoch_number = aux_model_config.global_epoch_number
    # aux_local_epoch_number = aux_model_config.local_epoch_number
    if non_iid_train:
        batch_start_id = 0
        batch_num = 1
        # ori_lr = aux_model_config.learning_rate2
        # decay = aux_model_config.learning_rate_decay2
        # aux_global_epoch_number = aux_model_config.global_epoch_number2
        # aux_local_epoch_number = aux_model_config.local_epoch_number2

  
        # if decay == LearningRateDecay.COSINE.value:
        #     lr = cosine_decay_lr(
        #         ori_lr,
        #         aux_epoch,
        #         aux_global_epoch_number,
        #     )
        # elif decay == LearningRateDecay.SMART_RECALL.value:
        #     lr = ori_lr
        #     extra_info[LearningRateDecay.SMART_RECALL.value] = {
        #         "window_size": 4,
        #         "pending_stage": (0 if dis_traininig_manage_info[b]["trained"] else 10),
        #         "decay_ratio": 0.3,
        #     }
        # else:
        #     lr = ori_lr
        # if aux_epoch >= 0:
        #     lr = 0.00002
    aux_model_caches = {m:[] for m in fl.split_parameters.global_m_set}
    for client_id in client_set:
        client: ClientMMULFED = fl.clients[client_id]
        returned_model, returned_info, training_time = client.aux_train(
            global_model_parameters=fl.model.state_dict(),
            global_aux_model_parameters_dict=aux_model_dict,
            global_epoch=epoch,
            local_epoch=fl.AC_local_epoch,
            log_interval=fl.log_interval,
            aux_model_config_dict=aux_model_config_dict,
            batch_start_id=batch_start_id,
            dis_threshold=fl.dis_threshold,
            batch_end_id=batch_start_id + batch_num,
        )
        
        if non_iid_train:
            batch_start_id += batch_num

        fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "TRAIN_FOR_AUX",  "epoch": epoch, "client_id": client_id, "info": returned_info, "training_time": training_time})

        for b in fl.split_parameters.global_m_set:
            if b not in returned_model:
                continue
            aux_model_caches[b].append(serialize_model(returned_model[b]["aux_model"]))


    for b in fl.split_parameters.global_m_set:
        if len(aux_model_caches[b]) == 0:
            continue
        aggregated_aux_model = aggregate_model_paras_in_list(aux_model_caches[b])
        deserialize_model(aux_model_dict[b], aggregated_aux_model)


def multimodal_unsupervised_learning_global_MMULAA(fl: FLAlgorithmInfo):
    assert fl.algorithm_name == FLFramework.MMULFEDAA.value
    global_m_set = fl.split_parameters.global_m_set

    alignment_loss_records = init_alignment_loss_records(global_m_set)
    alignment_pairs = list(alignment_loss_records.keys())
    dropout_records = []
    dropout_info = {}

    dis_traininig_manage_info = {}
    for m in global_m_set:
        dis_traininig_manage_info[m] = {"trained": False}
    

   
    
    dropout_L_dict = {pair: int(fl.aa_dropout_L * fl.global_epoch_number) for pair in alignment_pairs} 
    z_dict = {pair: -1 for pair in alignment_pairs} 
    ac_gap = fl.aa_AC_gap


    alignment_state_dict = {pair:1 for pair in alignment_pairs}

    for e in range(fl.global_epoch_number):
        selected_clients = sample_useful_clients({k:fl.clients[k].get_modality_set() for k in list(fl.clients.keys())}, fl.client_num, alignment_pairs)
        for client_id in selected_clients:
            client: Client = fl.clients[client_id]
            reconstruction_loss_info = client.eval_reconstruction(fl.model.state_dict())
            collect_alignment_loss_records(alignment_loss_records, reconstruction_loss_info["reconstruction_loss"], e, client_id, reconstruction_loss_info["sample_num"])
            fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "EVAL",  "epoch": e, "client_id": client_id, "training_time": reconstruction_loss_info['training_time']})
        
        alignment_loss_average = get_all_alignment_loss_average(alignment_loss_records)
        dropout_info = alignment_dropout_converged(alignment_loss_average, e, eps1= fl.aa_dropout_eps1, p_value_threshold1= fl.aa_dropout_pvalue1, eps2= fl.aa_dropout_eps2, p_value_threshold2= fl.aa_dropout_pvalue2, dropout_L_dict=dropout_L_dict )
        dropout_records.append(dropout_info)
       
        fl.save_func(alignment_loss_records, "alignment_loss_records")
        fl.save_func(dropout_records, "dropout_records")

        for pair in alignment_pairs:
            if alignment_state_dict[pair]==1 and dropout_info[pair]["con_drop"]:
                alignment_state_dict[pair]=0
                z_dict[pair] = int(((fl.global_epoch_number) - e)/2)

            elif alignment_state_dict[pair]==0 and (dropout_info[pair]["imp_drop"]  or z_dict[pair] == 0):
                alignment_state_dict[pair]=1
            
            if z_dict[pair] > 0:
                z_dict[pair] -= 1
        
        reserved_alignment_pairs_epoch = []
        for pair in alignment_pairs:
            if alignment_state_dict[pair]==1:
                reserved_alignment_pairs_epoch.append(pair)
        
        #reset if all pairs are dropped
        if len(reserved_alignment_pairs_epoch)==0:
            reserved_alignment_pairs_epoch = alignment_pairs
            for pair in alignment_pairs:
                alignment_state_dict[pair]=1
        
        #reselect according to undropped alignments
        reserved_clients = sample_useful_clients({k:fl.clients[k].get_modality_set() for k in selected_clients}, len(selected_clients), reserved_alignment_pairs_epoch)
        unselected_clients = list(filter(lambda k: k not in reserved_clients, list(fl.clients.keys())))
        reselected_clients = sample_useful_clients({k:fl.clients[k].get_modality_set() for k in unselected_clients}, fl.client_num - len(reserved_clients), reserved_alignment_pairs_epoch)

        final_selected_clients = reserved_clients + reselected_clients

        if fl.with_aux == 1:
            ac_gap -= 1
            clients_for_aux = sample_clients_normal(list(filter(lambda k: len(fl.clients[k].get_modality_set())>1, list(fl.clients.keys()))), fl.client_num)
            supervised_global_training_for_discriminator(clients_for_aux,fl,{"epoch": e})

        model_params_cache_encoders = [[] for i in range(fl.channel_num)]
        model_params_cache_decoders = [[] for i in range(fl.channel_num)]

        for client_id in final_selected_clients:
            client: Client = fl.clients[client_id]

            res = client.train(
                global_model_parameters=fl.model.state_dict(),
                global_epoch=e,
                local_epoch=fl.local_epoch_number,
                log_interval=fl.log_interval,
                selected_pairs=reserved_alignment_pairs_epoch
                )
            client_model,train_loss,train_acc,val_loss,val_acc,training_encoder_modalities,training_decoder_modalities, training_time = res['returned_model'],res['train_losses'],res['train_acc'],res['val_losses'],res['val_acc'], res['training_encoder_modalities'], res['training_decoder_modalities'], res['training_time']
            fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "COMPUTATION_TIME", "training_time": training_time, "epoch": e, "client_id":client_id, "reserved_alignment_pairs_epoch":reserved_alignment_pairs_epoch, "client_modality_set": client.get_modality_set() , "type":"normal"})

            collect_encoders_decoders(
                encoder_list=client_model.encoder_list,
                model_params_cache_encoders=model_params_cache_encoders,
                decoder_list=client_model.decoder_list,
                model_params_cache_decoders=model_params_cache_decoders,
                client_M_set_encoders=training_encoder_modalities,
                client_M_set_decoders=training_decoder_modalities,
                global_M_set=fl.split_parameters.global_m_set,
            )
        
        aggregate_collected_encoders_decoders(
            encoder_list=fl.model.encoder_list,
            model_params_cache_encoders=model_params_cache_encoders,
            decoder_list=fl.model.decoder_list,
            model_params_cache_decoders=model_params_cache_decoders,
            global_M_set=fl.split_parameters.global_m_set,
        )

        if fl.with_aux == 1:
            if ac_gap == 0:
                clients_with_aux = sample_clients_normal(list(filter(lambda k: len(fl.clients[k].get_modality_set())< len(fl.split_parameters.global_m_set), list(fl.clients.keys()))), fl.client_num)
                unsupervised_global_training_with_discriminator(clients_with_aux, fl, {"epoch":e})
                ac_gap = fl.aa_AC_gap

        if fl.dataset_name != "URFALL":
            for client_id in fl.clients.keys():
                client: Client = fl.clients[client_id]
                reconstruction_loss_client = client.eval_reconstruction(fl.model.state_dict())["reconstruction_loss"]
                fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "RECONSTRUCTION_CLIENT", "reconstruction_loss": reconstruction_loss_client, "epoch": e, "client_id": client_id})
        gobal_reconstruction_loss = fl.global_client.eval_reconstruction(fl.model.state_dict())["reconstruction_loss"]
        fl.log_func({DataAnalysisDimensions.LOG_TYPE.value: "RECONSTRUCTION", "reconstruction_loss": gobal_reconstruction_loss, "epoch": e, "alignment_pairs_epoch":reserved_alignment_pairs_epoch})
