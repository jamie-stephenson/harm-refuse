from harm_refuse.utils.data import get_dataset 

from datasets import DatasetDict
import torch

def _find_mean_resid(
    ds: DatasetDict,
    ids: List[int],
):  
    centers = {}
    for name, data in ds.items():
        acts = torch.stack([t for t in data["resid"]])
        centers[name] = acts[:, :, ids].mean(dim = 0)

    return centers

def cluster(config):
    model = get_model(config)
    ids = get_ids(model)
    ds = get_dataset()
    cluster_centers = _find_mean_resid(ds, ids)
