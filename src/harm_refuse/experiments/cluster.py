from harm_refuse.utils.data import get_dataset 
from harm_refuse.utils.config import Config 

from nnsight import LanguageModel
import torch
from torch.nn.functional import normalize, cosine_similarity as cossim 
    
def cluster(
    config: Config
):
    model = config.get_model()
    ids = config.get_ids(model)
    ds = get_dataset(config.get_dataset_path(), model)

    # (n_clusters, n_layers, n_ids, d_model)
    cluster_centers = torch.stack([
        ds[name]["resid"][:, :, ids].mean(dim = 0)
        for name in ["harmful_refused", "harmless_accepted"]
    ]) 

    # (n_mistakes, batch_size, n_layers, n_ids, d_model)
    mistakes = torch.stack([
        ds[name]["resid"][:, :, ids]
        for name in ["harmful_accepted", "harmless_refused"]
    ]) 
    
    # (n_clusters, n_mistakes, batch_size, n_layers, n_ids)
    cossims = cossim(
        cluster_centers[:, None, None],
        mistakes[None],
        dim = -1,
    )

    cossims2 = torch.einsum(
        "c l i d, m b l i d -> c m b l i",
        normalize(cluster_centers, dim=-1),
        normalize(mistakes, dim=-1),
    )

    # (n_mistakes, batch_size, n_layers, n_ids)
    cossim_diff = cossims[0] - cossims[1]

    assert torch.allclose(cossims, cossims2)
