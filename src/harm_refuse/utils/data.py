"""
Utilities for creating datasets for all experiments
"""
from .inference import run_inference
from .refuse import is_refused
from .config import get_dataset_path

from nnsight import LanguageModel
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, load_from_disk

from typing import Dict
from pathlib import Path

def _reformat(
    ds: Dataset,
    model: LanguageModel,
    is_harmful: bool,
    prompt_key: str,
    category_key: str,
    source: str,
    batch_size: int = 1,
) -> Dataset:
    """Normalises datasets to format we need for experiments."""
    rows = []
    for batch in ds.batch(batch_size):
        prompts = batch[prompt_key]
        responses, resid = run_inference(prompts, model)
        rows.extend([
            {
                "prompt": p,
                "source": source,
                "category": c,
                "source_id": f"{source.lower()}_{i}",
                "is_harmful": is_harmful,
                "is_refused": is_refused(r),
                "resid": h,
            } 
            for i, (p, r, h, c) in enumerate(
                zip(
                    prompts, 
                    responses, 
                    resid, 
                    batch.get(category_key, ""),
                )
            )
        ])

    return Dataset.from_list(rows)

def advbench(model: LanguageModel) -> Dataset:
    """
    AdvBench harmful behaviors.
    HF hub: walledai/AdvBench (train split, columns include 'prompt').  MIT license.
    """
    ds = load_dataset("walledai/AdvBench")["train"]  # columns: ['prompt', 'target'] etc.
    return _reformat(
        ds,
        model,
        is_harmful=True,
        prompt_key="prompt",
        category_key=None,
        source="AdvBench",
    )

def jbb(model: LanguageModel) -> Dict[str, Dataset]:
    """
    JailbreakBench behaviors: has 'harmful' and 'benign' splits.
    Use 'Goal' as the prompt; 'Category' is available.  MIT license.
    """
    data = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful_split = data["harmful"]
    benign_split  = data["benign"]

    harmful = _reformat(
        ds=harmful_split,
        model=model,
        is_harmful=True,
        prompt_key="Goal",
        category_key="Category",
        source="JBB",
    )

    benign = _reformat(
        ds=benign_split,
        model=model,
        is_harmful=False,
        prompt_key="Goal",
        category_key="Category",
        source="JBB",
    )

    return concatenate_datasets([harmful, benign]) 

def sorry_bench(model: LanguageModel) -> Dataset:
    """
    SORRY-Bench unsafe instructions across 44 categories (+ mutations).
    You must accept the license on the hub and be logged in.
    """

    ds = load_dataset(
        "sorry-bench/sorry-bench-202503",
        token="", 
        split="train",
    ).map(
        # Prompt is wrapped in list that is always length 1. 
        lambda e: {"turns": e["turns"][0]} 
    )

    return _reformat(
        ds,
        model,
        is_harmful=True,
        prompt_key="turns",
        category_key="category",
        source="SORRY-Bench",
    )


def xstest(model: LanguageModel) -> Dict[str, Dataset]:
    """
    XSTest includes 250 safe + 200 unsafe prompts (labels 'safe'/'unsafe').
    Use 'prompt' as text; include 'type' or 'focus' in category if present.  CC-BY-4.0.
    """
    ds = load_dataset("walledai/XSTest")["test"]  # columns include: prompt, label ('safe'/'unsafe'), type, focus, note
    safe = ds.filter(lambda x: x["label"] == "safe")
    unsafe = ds.filter(lambda x: x["label"] == "unsafe")

    safe = _reformat(
        ds=safe,
        model=model,
        is_harmful=False,
        prompt_key="prompt",
        category_key=None,
        source="XSTest",
    )

    unsafe = _reformat(
        ds=unsafe,
        model=model,
        is_harmful=True,
        prompt_key="prompt",
        category_key=None,
        source="XSTest",
    )
    
    return concatenate_datasets([safe, unsafe])

def alpaca(model: LanguageModel) -> Dataset:
    """
    Alpaca instruction-tuning data (harmless). Use 'instruction' + optional 'input' to form a single prompt.
    Prefer the popular cleaned variant: yahma/alpaca-cleaned (CC-BY-4.0).
    """
    ds = load_dataset("yahma/alpaca-cleaned")["train"]  # columns: instruction, input, output
    
    ds = ds.map(
        lambda e: {
            "prompt": e["instruction"] + (("\n" + e["input"]) if e.get("input") else "")
        },
        remove_columns=["instruction", "input"]
    )

    return _reformat(
        ds,
        model,
        is_harmful=False,
        prompt_key="prompt",
        category_key=None,
        source="Alpaca",
    )

def _build_dataset(
    path: Path,
    model: LanguageModel,
) -> DatasetDict:
    components = [
        advbench,
        jbb,
        sorry_bench,
        xstest,
        alpaca,
    ]

    base = concatenate_datasets([
        ds(model) for ds in components
    ])

    combos = [
        ("harmless_refused",  False, True),
        ("harmless_accepted", False, False),
        ("harmful_refused",   True,  True),
        ("harmful_accepted",  True,  False),
    ]

    out = {
        name: base.filter(
            lambda x: x["is_harmful"] == h and x["is_refused"] == r
        )
        for name, h, r in combos
    }

    ds = DatasetDict(out)
    ds.save_to_disk(path)

    return ds

def get_dataset(
    model: LanguageModel,
    n_samples: int = 100,
    strict: bool = True,
    seed: int = 90,
) -> DatasetDict:
    """
    Get `DatasetDict` consisting of `n_samples` from each prompt type: 
    - Accepted harmless
    - Accepted harmful
    - Refused harmless
    - Refused harmful

    Each prompt has the `model`'s chat template applied  to it.
    """
    path = get_dataset_path(model)
    if path.exists():
        ds = load_from_disk(path)
    else:
        ds = _build_dataset(path, model)

    sampled_splits = {}
    for name, split in ds.items():
        if strict and len(split) < n_samples:
            raise ValueError(
                f"Not enough rows for {name}: have {len(split)}, need {n_samples}."
            )
        shuffled = split.shuffle(seed)
        sampled = shuffled.select(range(min(n_samples, len(shuffled))))
        sampled_splits[name] = sampled

    return DatasetDict(sampled_splits)

