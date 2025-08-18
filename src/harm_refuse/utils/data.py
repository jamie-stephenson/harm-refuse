"""
Utilities for creating datasets for all experiments
"""
from nnsight import LanguageModel
from .load.prompts import load_prompts

def build_dataset(
    model: LanguageModel,
    n_samples: int = 400,
):
    dataset = load_prompts()
    chats = [
        {"role": "user", "content": prompt}
        for prompt in dataset["prompt"]
    ] # TODO, apply to dataset instead of making new object

    templated_prompts = model.tokenizer.apply_chat_template(chats, tokenize=False)
    print(templated_prompts)




