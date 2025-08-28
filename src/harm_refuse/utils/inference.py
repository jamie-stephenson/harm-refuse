"""
Utilities for model inference.
"""
from nnsight import LanguageModel
from torch import Tensor

from typing import Tuple, List

def _tokenize(prompts: List[str], model: LanguageModel) -> Tensor:
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return model.tokenizer.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ) # pyright: ignore[reportReturnType]

def run_inference(
    prompts: List[str],
    model: LanguageModel,
) -> Tuple[List[str], Tensor]:
    tokens = _tokenize(prompts, model)
    with model.generate(tokens, max_new_tokens=256):
        # Save the residual stream after *one* forward pass
        resid = model.model.layers.output.save()

        # Save output tokens after *all* forwards passes
        response_tokens = model.generator.output.save()

    responses = model.tokenizer.batch_decode(
        response_tokens,
        skip_special_tokens=True,
    )

    return responses, resid
