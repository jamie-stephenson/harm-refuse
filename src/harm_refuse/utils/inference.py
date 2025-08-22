"""
Utilities for model inference.
"""
from nnsight import LanguageModel
from torch import Tensor

from typing import Tuple

def _tokenize(prompt:str, model: LanguageModel) -> Tensor:
    """
    Applies chat template and then tokenizes.
    """
    return model.tokenizer.apply_chat_template(
        conversation = [{
            "role": "user",
            "content": prompt
        }],
        add_generation_prompt=True,
        return_tensors="pt",
    ) # pyright: ignore[reportReturnType]

def run_inference(
    prompt: str, 
    model: LanguageModel
) -> Tuple[Tensor, Tensor, bool]:
    tokens = _tokenize(prompt, model)
    with model.generate(tokens, max_new_tokens=256):
        # Save the residual stream after *one* forward pass
        resid = model.transformer.h.output.save()

        # Save output tokens after *all* forwards passes
        response_tokens = model.generator.output.save()

    response_str = model.tokenizer.decode(
        response_tokens.value[0].tolist()[len(tokens):],
        skip_special_tokens=True,
    )

    pass
