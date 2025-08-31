from harm_refuse.utils.data import get_dataset 
from harm_refuse.utils.model import get_model, get_ids 

import torch
from torch.nn.functional import normalize, cosine_similarity as cossim 

def cluster(config):
    model = get_model(config)
    ids = get_ids(model)
    ds = get_dataset(model)

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

    assert torch.allclose(cossims, cossims2)
