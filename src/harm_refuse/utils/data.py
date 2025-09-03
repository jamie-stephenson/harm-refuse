"""
Utilities for creating datasets for all experiments
"""
from .inference import run_inference
from .refuse import is_refused

from nnsight import LanguageModel
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
    load_dataset,
    Array3D,
)
from tqdm.auto import tqdm
import numpy as np

from math import ceil
from pathlib import Path

def _reformat(
    ds: Dataset,
    model: LanguageModel,
    is_harmful: bool,
    prompt_key: str,
    category_key: str,
    source: str,
    batch_size: int = 128,
) -> Dataset:
    """Normalises datasets to format we need for experiments."""
    rows = []
    batched_ds_with_pbar = tqdm(
        ds.iter(batch_size),
        total=ceil(len(ds)/batch_size),
        unit="batch",
        desc=f"{source}",
    )

    for batch in batched_ds_with_pbar:
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
                    batch.get(category_key, [""]*len(prompts)),
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

def jbb(model: LanguageModel) -> Dataset:
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


def xstest(model: LanguageModel) -> Dataset:
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

def _build_base_dataset(
    path: Path,
    model: LanguageModel,
) -> Dataset:
    components = [
        advbench,
        jbb,
        # sorry_bench,
        xstest,
        # alpaca,
    ]

    base = concatenate_datasets([
        ds(model) for ds in components
    ])

    # shortest sequence length
    s_min = min(min(len(seq) for seq in ex) for ex in base["resid"])

    def trim_batch(batch):
        return {"resid": [[seq[-s_min:] for seq in ex] for ex in batch["resid"]]}

    base_trimmed = base.map(
        trim_batch,
        batched=True,
        batch_size=256,
        num_proc=1,
        load_from_cache_file=False,
    )

    L = len(base_trimmed[0]["resid"])
    D = len(base_trimmed[0]["resid"][0][0])
    base_trimmed = base_trimmed.cast_column("resid", Array3D((L, s_min, D), "float32"))

    base_trimmed.save_to_disk(path)

    return base_trimmed

def get_dataset(
    path: Path,
    model: LanguageModel,
    n_samples: int = 100,
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
    if path.exists():
        ds = load_from_disk(path)
    else:
        ds = _build_base_dataset(path, model)

    h = np.array(ds["is_harmful"], bool)
    r = np.array(ds["is_refused"], bool)

    masks = {
        "harmless_refused": (~h & r),
        "harmless_accepted": (~h & ~r),
        "harmful_refused": (h & r),
        "harmful_accepted": (h & ~r)
    }
   
    rng = np.random.default_rng(seed)
    splits = {}
    for k, m in masks.items():
        if not np.any(m):
            raise ValueError(f"Split '{k}' is empty.")
        elif n_samples > m.sum():
            raise ValueError(
                f"Not enough examples in split '{k}' ({m.sum()}<{n_samples})."
            )
        else:
            splits[k] = ds.select(
                rng.choice(np.where(m)[0], size=n_samples, replace=False).tolist()
            ) 

    ds = DatasetDict(splits)
    ds.set_format("torch", columns=["resid"])

    return ds

