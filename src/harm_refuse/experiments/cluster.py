from nnsight.modeling.language import LanguageModel
import torch

from typing import List

def find_mean_resid(
    model: LanguageModel,
    prompts: List[str],
):  
    """
    Given a batch of `b` prompts, perform a single forward pass on all of
    them and find the mean of the residual stream at each layer and token
    position.
    """
    batch = model.tokenizer(prompts, return_tensors="pt", padding=True)
    mask = get_mask() 

    with model.trace(batch):
        resid = model.transformer.h.output.save()

    acts = torch.stack(
        [layer[0] for layer in resid], # pyright: ignore[reportArgumentType]
        dim = 1,
    ) # [b, l, s, d]

    mean_acts = acts[:, :, mask].mean(dim = 0)

    return mean_acts
