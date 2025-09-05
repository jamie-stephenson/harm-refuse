"""
Utilities for model inference.
"""
from nnsight import LanguageModel
import torch
from torch import Tensor


def _tokenize(prompts: list[str], model: LanguageModel) -> Tensor:
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return model.tokenizer.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ) # pyright: ignore[reportReturnType]

def run_inference(
    prompts: list[str],
    model: LanguageModel,
    ids: list[int],
) -> tuple[list[str], Tensor]:
    tokens = _tokenize(prompts, model)
    layers = model.model.layers
    n_layers = len(layers)
    with torch.inference_mode():
        with model.generate(tokens, max_new_tokens=256, remote=True):
            # Save the residual stream after *one* forward pass
            resid_list = [layers[i].output[:, ids] for i in range(n_layers)].save()

            # Save output tokens after *all* forwards passes
            response_tokens = model.generator.output.save()

    resid = torch.stack(resid_list, dim=1).cpu()
    responses = model.tokenizer.batch_decode(
        response_tokens,
        skip_special_tokens=False,
    )

    return responses, resid
