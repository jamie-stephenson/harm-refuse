from harm_refuse.utils.data import get_dataset 
from harm_refuse.utils.config import Config 

import torch
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import numpy as np

import math


def cluster(
    config: Config
):
    model = config.get_model()
    ids, tail_len = config.get_ids(model.tokenizer)
    ds = get_dataset(
        config.paths.data, 
        model,
        ids,
        n_samples=config.n_samples,
    )

    # (n_clusters, n_layers, n_ids, d_model)
    cluster_centers = torch.stack([
        ds[name][:]["resid"].mean(dim = 0) 
        for name in ["harmful_refused", "harmless_accepted"]
    ])

    # (n_mistakes, batch_size, n_layers, n_ids, d_model)
    mistakes = torch.stack([
        ds[name][:]["resid"]
        for name in ["harmful_accepted", "harmless_refused"]
    ]) 
    
    # (n_clusters, n_mistakes, batch_size, n_layers, n_ids)
    cossims = torch.einsum(
        "c l i d, m b l i d -> c m b l i",
        normalize(cluster_centers, dim=-1),
        normalize(mistakes, dim=-1),
    )

    # Positive cossim diff means more similarity
    # with refused than accepted.
    # (n_mistakes, batch_size, n_layers, n_ids)
    cossim_diff = cossims[0] - cossims[1]

    plot_graphs(cossim_diff, tail_len, ids)
    plt.show()

def plot_graphs(
    cossim_diff: torch.Tensor,
    tail_len: int,
    ids: list[int],
    ci_multiplier: float = 1.96,
):
    """Plot layer-wise cosine-similarity deltas with 95% CI bands.

    Expects `similarity_delta` shaped (2, batch_size, n_layers, n_ids), where the
    first axis corresponds to ["harmful_accepted", "harmless_refused"].
    """
    # Note: here n_ids <= config.n_ids as there may be an overlap
    # between pre t_inst tokens and pre t_post tokens.
    n_mistakes, batch_size, n_layers, n_ids = cossim_diff.shape
    if cossim_diff.ndim != 4 or n_mistakes != 2:
        raise ValueError("Expected shape (2, batch_size, n_layers, n_ids).")

    # Reduce over batch dimension -> (2, n_layers, n_ids)
    mean = cossim_diff.mean(dim=1)
    if batch_size > 1:
        std = cossim_diff.std(dim=1, unbiased=True)
        sem = std / math.sqrt(batch_size)
    else:
        sem = torch.zeros_like(mean)

    layers = np.arange(n_layers)
    mean_np = mean.detach().cpu().numpy()
    sem_np = sem.detach().cpu().numpy()

    mistake_labels = ["harmful_accepted", "harmless_refused"]

    def token_label(token_idx: int) -> str:
        # Where we are relative to t_inst and t_post
        t_inst_idx = token_idx + tail_len + 1
        if t_inst_idx < 0:
            return f"t_inst - {-t_inst_idx}"
        if t_inst_idx == 0:
            return "t_inst"
        if token_idx < -1:
            return f"t_post - {-token_idx-1}"
        if token_idx == -1:
            return "t_post"
        raise ValueError(
            f"Error indexing tokens relative to t_inst({-tail_len-1}) and t_post(-1). "
            f"Token ids relative to end of sequence: {ids}."
        )

    fig, axes = plt.subplots(
        1, n_ids, figsize=(10, 4), sharey=True, constrained_layout=True
    )
    # Ensure iterable axes
    axes = np.atleast_1d(axes)

    for pos_idx, ax in enumerate(axes):
        for mistake_idx, mistake in enumerate(mistake_labels):
            y = mean_np[mistake_idx, :, pos_idx]
            e = sem_np[mistake_idx, :, pos_idx]
            line, = ax.plot(layers, y, label=mistake, linewidth=2)
            ax.fill_between(layers, y - ci_multiplier * e, y + ci_multiplier * e,
                            alpha=0.2, color=line.get_color())

        ax.set_title(token_label(ids[pos_idx]))
        ax.set_xlabel("layer")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("mean cossim diff")
    axes[-1].legend(title="line")
    return fig, axes
