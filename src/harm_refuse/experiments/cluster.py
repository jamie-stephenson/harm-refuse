from harm_refuse.utils.data import get_dataset 
from harm_refuse.utils.config import Config 

import torch
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

import math


def cluster(
    config: Config
):
    model = config.get_model()
    ds = get_dataset(
        config.paths.data, 
        model,
        n_samples=config.n_samples,
    )

    ids = config.get_ids(model.tokenizer)

    # (n_clusters, n_layers, n_ids, d_model)
    cluster_centers = torch.stack([
        ds[name][:]["resid"][:, :, ids].mean(dim = 0) 
        for name in ["harmful_refused", "harmless_accepted"]
    ])

    # (n_mistakes, batch_size, n_layers, n_ids, d_model)
    mistakes = torch.stack([
        ds[name][:]["resid"][:, :, ids]
        for name in ["harmful_accepted", "harmless_refused"]
    ]) 
    
    # (n_clusters, n_mistakes, batch_size, n_layers, n_ids)
    cossims = torch.einsum(
        "c l i d, m b l i d -> c m b l i",
        normalize(cluster_centers, dim=-1),
        normalize(mistakes, dim=-1),
    )

    # (n_mistakes, batch_size, n_layers, n_ids)
    cossim_diff = cossims[0] - cossims[1]

    plot_tensor_lines_with_ci(cossim_diff)
    plt.show()

def plot_tensor_lines_with_ci(t: torch.Tensor, z: float = 1.96):
    """
    Plot two panels from a tensor of shape (2, batch, n_layers, n_ids).
      • Left panel: last-dim index 0
      • Right panel: last-dim index 1
    Each panel shows two lines for first-dim index 0 and 1.
    Y = batch mean; shaded band = z * standard error (std/sqrt(batch)).
    Returns (fig, axes).
    """
    n_mistakes, batch, n_layers, n_ids = t.shape
    if t.ndim != 4 or n_mistakes != 2:
        raise ValueError("Expected shape (2, batch_size, n_layers, n_ids).")

    # Reduce over batch dimension
    means = t.mean(dim=1)  # (2, n_layers, n_ids)
    if batch > 1:
        std = t.std(dim=1, unbiased=True)
        se = std / math.sqrt(batch)
    else:
        se = torch.zeros_like(means)

    x = torch.arange(n_layers)

    def _np(x):
        return x.detach().cpu().numpy()

    fig, axes = plt.subplots(1, n_ids, figsize=(10, 4), sharey=True, constrained_layout=True)
    for last_idx, ax in enumerate(axes):
        for first_idx in (0, 1):
            y = means[first_idx, :, last_idx]
            e = se[first_idx, :, last_idx]
            ln, = ax.plot(_np(x), _np(y), label=f"dim0={first_idx}", linewidth=2)
            ax.fill_between(_np(x), _np(y - z * e), _np(y + z * e),
                            alpha=0.2, color=ln.get_color())
        ax.set_title(f"dim3={last_idx}")
        ax.set_xlabel("layer index")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("batch mean")
    axes[1].legend(title="line")
    return fig, axes
