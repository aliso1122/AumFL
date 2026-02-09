
from datasets import load_dataset, load_from_disk
from data_generating.utils import get_datadir
import os

cache_dir, _ = get_datadir("QILIN", "a")
def load_Qilin_data(subset_name, download=False):
    assert subset_name in ('notes', 'recommendation_train', 'recommendation_test', 'user_feat')
    subset_dir = f"{cache_dir}/{subset_name}"
    if download or not os.path.isdir(subset_dir):
        subset = load_dataset("THUIR/Qilin", subset_name)
        subset.save_to_disk(subset_dir)
    subset = load_from_disk(subset_dir)
    return subset['train']