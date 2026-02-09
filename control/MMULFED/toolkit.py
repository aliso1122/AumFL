from control.globalside_toolkit import sample_clients_normal
from control.Enums import ParametersForDataSplitType
import numpy as np

from scipy import stats

def trend_slope(loss_arr, window=20):
    if len(loss_arr) < window:
        return None, None
    
    ll = len(loss_arr)
    recent = loss_arr[ll-window:ll]
    recent_norm = (recent - np.mean(recent)) / (np.std(recent) + 1e-8)
    x = np.arange(window)
    
    slope, _, _, p_value, _ = stats.linregress(x, recent_norm)
    return slope, p_value

def sample_useful_clients(client_k_modalityset: dict, sample_num: int, pairs:list):
    def dicar_intersection(modalityset, pairs):
        dicar_sets = []
        for a in modalityset:
            for b in modalityset:
                dicar_sets.append(f"{a}-{b}")
        for s in dicar_sets:
            if s in pairs:
                return True
        return False

    useful_clients_keys = list(filter(lambda k: dicar_intersection(client_k_modalityset[k], pairs), list(client_k_modalityset.keys())))

    return sample_clients_normal(useful_clients_keys, sample_num)

def sample_clients_MMULFED(
    client_group_dict: dict,
    global_epoch_number: int,
    client_num: int,
):
    group_num = len(client_group_dict)
    group = client_group_dict[global_epoch_number % group_num]
    return sample_clients_normal(group, client_num)


def sample_clients_by_modalities(
    a: int,
    b: int,
    n_C_ab: int,
    n_C_a_not_b: int,
    n_C_b_not_a: int,
    parameters: ParametersForDataSplitType,
):
    def sample_a_subset(U: list, n_subset: int):
        n_subset = min(n_subset, len(U))
        idxes = np.random.choice(list(range(len(U))), n_subset, replace=False)
        n_subset = [U[idx] for idx in idxes]
        return n_subset

    U_a = parameters.modality_2_clients[a]

    U_b = parameters.modality_2_clients[b]
    # print("U", U_a, U_b)
    U_ab = list(set(U_a) & set(U_b))
    U_a_not_b = list(set(U_a) - set(U_ab))
    U_b_not_a = list(set(U_b) - set(U_ab))

    C_ab = sample_a_subset(U_ab, n_C_ab)
    C_a_not_b = sample_a_subset(U_a_not_b, n_C_a_not_b)
    C_b_not_a = sample_a_subset(U_b_not_a, n_C_b_not_a)

    C_ab = list(set(C_ab))
    # print(parameters.modality_2_clients)
    return (C_ab, C_a_not_b, C_b_not_a)
