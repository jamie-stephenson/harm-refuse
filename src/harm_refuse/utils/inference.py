"""
Utilities for loading and using models.
"""
from nnsight import LanguageModel
from torch import Tensor

from typing import Tuple

def _tokenize(prompt:str, model: LanguageModel) -> Tensor:
    """
    Applies chat template and then tokenizes.
    """
    return model.tokenizer.apply_chat_template(
        converstaion = [{
            "role": "user",
            "content": prompt
        }],
        add_generation_prompt=True,
    )

def run_inference(
    prompt: str, 
    model: LanguageModel
) -> Tuple[Tensor, Tensor, bool]:
    tokens = _tokenize(prompt, model)
    with model.trace(tokens):

    pass
