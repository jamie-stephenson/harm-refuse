from nnsight import LanguageModel
from transformers import AutoTokenizer
from hydra.core.config_store import ConfigStore

from dataclasses import dataclass
from typing import Any
from warnings import warn
from pathlib import Path

@dataclass
class Paths:
    model: Path
    data: Path
    out: Path

@dataclass
class Config:
    name: str
    paths: Paths
    device: str = "auto"
    n_ids: int = 1

    def get_model(self) -> LanguageModel:
        return LanguageModel(
            self.model_path,
            device_map=self.device,
        )

    def chat_tail_len(self,tokenizer: Any = None) -> int:
        has_chat = hasattr(tokenizer, "apply_chat_template")
        if tokenizer and not has_chat:
            warn(
                f"""
                Cannot infer chat tail length from input `tokenizer` of type {type(tokenizer)}.
                Defaulting to using `AutoTokenizer` and the provided Hugging Face model path.
                Double check that the tokenizer that you are passing as input does actually
                support an Assitant/Chat/Instruct use case.
                """
            )

        # Resolve tokenizer
        tok = tokenizer if has_chat else AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        mark = "\uE000\uE001\uE002"
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": mark}],
            add_generation_prompt=True,
            tokenize=False,
        )
        suffix = rendered.split(mark, 1)[1]
        return len(tok(suffix, add_special_tokens=False)["input_ids"])

    def get_ids(self, tokenizer: Any = None) -> list[int]:
        """
        Gets the position ids (at which we run the experiment) based on `n_ids`. 

        `n_ids == 1`: only fetch `t_inst` and `t_post`.
        `n_ids > 1`: fetch `t_inst, t_inst - 1, ... t_inst - n_ids + 1` and similar 
        for t_post.

        We are flexible with the object we infer `t_inst` from for development purposes. 
        """
        tail_len = self.chat_tail_len(tokenizer)
        return list({
            -i-1 
            for j in range(0, self.n_ids)
            for i in (j, j-tail_len)
        })


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="paths", name="paths", node=Paths)
