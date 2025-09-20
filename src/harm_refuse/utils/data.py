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
    Features,
    Value,
    Array3D,
)
import numpy as np

from pathlib import Path


def _reformat(
    ds: Dataset,
    model: LanguageModel,
    is_harmful: bool,
    prompt_key: str,
    category_key: str,
    source: str,
    remote: bool,
    ids: list[int],
    batch_size: int = 128,
) -> Dataset:
    """Normalises datasets to format we need for experiments."""

    prompts = list(ds[prompt_key])
    responses, resid = run_inference(prompts, model, ids, batch_size, remote)

    L = model.config.num_hidden_layers
    S = len(ids)
    D = model.config.hidden_size

    cols = {
        "prompt": [item[prompt_key] for item in ds],
        "source": [source]*len(ds),
        "category": [item.get(category_key, "") for item in ds],
        "is_harmful": [is_harmful]*len(ds),
        "is_refused": [is_refused(r) for r in responses],
        "resid": [h for h in resid],
    }

    feats = Features({
        "prompt": Value("string"),
        "source": Value("string"),
        "category": Value("string"),
        "is_harmful": Value("bool"),
        "is_refused": Value("bool"),
        "resid": Array3D((L, S, D), "float32"),
    })

    return Dataset.from_dict(cols, features=feats)

def advbench(model: LanguageModel, **kwargs) -> Dataset:
    """
    AdvBench harmful behaviors.
    HF hub: walledai/AdvBench (train split, columns include 'prompt').  MIT license.
    """
    ds = load_dataset("walledai/AdvBench", split="train")  # columns: ['prompt', 'target'] etc.
    return _reformat(
        ds,
        model,
        is_harmful=True,
        prompt_key="prompt",
        category_key=None,
        source="AdvBench",
        **kwargs,
    )

def jbb(model: LanguageModel, **kwargs) -> Dataset:
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
        **kwargs,
    )

    benign = _reformat(
        ds=benign_split,
        model=model,
        is_harmful=False,
        prompt_key="Goal",
        category_key="Category",
        source="JBB",
        **kwargs,
    )

    return concatenate_datasets([harmful, benign]) 

def sorry_bench(model: LanguageModel, **kwargs) -> Dataset:
    """
    SORRY-Bench unsafe instructions across 44 categories (+ mutations).
    You must accept the license on the hub and be logged in.
    """

    ds = load_dataset(
        "sorry-bench/sorry-bench-202503",
        token="", 
        split="train[:1000]",
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
        batch_size=16,
        **kwargs,
    )


def xstest(model: LanguageModel, **kwargs) -> Dataset:
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
        **kwargs,
    )

    unsafe = _reformat(
        ds=unsafe,
        model=model,
        is_harmful=True,
        prompt_key="prompt",
        category_key=None,
        source="XSTest",
        **kwargs
    )
    
    return concatenate_datasets([safe, unsafe])

def alpaca(model: LanguageModel, **kwargs) -> Dataset:
    """
    Alpaca instruction-tuning data (harmless). Use 'instruction' + optional 'input' to form a single prompt.
    Prefer the popular cleaned variant: yahma/alpaca-cleaned (CC-BY-4.0).
    """
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # columns: instruction, input, output
    
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
        **kwargs,
    )

def _build_from_local(
    model: LanguageModel,
    ids: list[int],
    remote: bool,
    data_dir: Path, 
) -> Dataset:
    """Load newline-delimited JSON prompts from `data/` and normalise via `_reformat`."""

    import json

    harm_overrides = {
        "alpaca_data_instruction.json": False,
        "example_test_harmless.json": False,
        "xstest-harmless.json": False,
    }

    formatted_datasets = []
    for path in sorted(data_dir.glob("*.json")):
        files_to_skip = ["GPTFuzzer-50-adv", "example_test", "human-seed"]
        if any(n in str(path) for n in files_to_skip):
            continue

        print("Processing file: ", path)
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i > 500:
                    break
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                prompt = obj.get("bad_q") or obj.get("instruction")
                if not prompt:
                    continue

                category = (
                    obj.get("category")
                    or obj.get("Category")
                    or obj.get("Subcategory")
                    or obj.get("focus")
                    or obj.get("type")
                    or obj.get("dataset")
                    or obj.get("note")
                    or ""
                )

                records.append({"prompt": prompt, "category": category or ""})

        if not records:
            continue

        is_harmful = harm_overrides.get(path.name)
        if is_harmful is None:
            lowered = path.stem.lower()
            if "harmless" in lowered or "alpaca" in lowered:
                is_harmful = False
            else:
                is_harmful = True

        ds = Dataset.from_list(records)
        formatted = _reformat(
            ds=ds,
            model=model,
            ids=ids,
            is_harmful=is_harmful,
            prompt_key="prompt",
            category_key="category",
            source=path.stem,
            remote=remote,
        )

        formatted_datasets.append(formatted)

    if not formatted_datasets:
        raise ValueError(f"No JSON prompts found in {data_dir}")

    if len(formatted_datasets) == 1:
        return formatted_datasets[0]

    return concatenate_datasets(formatted_datasets)

def _build_base_dataset(
    path: Path,
    model: LanguageModel,
    ids: list[int],
    local_data: bool,
    remote: bool,
) -> Dataset:
    if local_data:
        base = _build_from_local(model, ids, remote, Path("./local_data/"))
    else:
        components = [
            advbench,
            jbb,
            sorry_bench,
            xstest,
            alpaca,
        ]

        base = concatenate_datasets([
            ds(model, ids=ids, remote=remote) for ds in components
        ])
    
    # idk if all models expose their structure like this
    # but all the ones for this project do.
    L = model.config.num_hidden_layers
    S = len(ids)
    D = model.config.hidden_size

    base = base.cast_column(
        "resid",
        Array3D((L, S, D),"float32")
    )

    base.save_to_disk(path)

    return base

def get_dataset(
    path: Path,
    model: LanguageModel,
    n_samples: int,
    local_data: bool,
    remote: bool,
    seed: int = 91,
    ids: list[int] | None = None,
) -> DatasetDict:
    """
    Get `DatasetDict` consisting of `n_samples` from each prompt type: 
    - Accepted harmless
    - Accepted harmful
    - Refused harmless
    - Refused harmful

    Each prompt has the `model`'s chat template applied  to it.

    `local`: whether to use data saved in `local_data`.
    """
    if path.exists():
        # TODO: check that ids given match dataset saved on disk
        ds = load_from_disk(path)
    elif ids:
        ds = _build_base_dataset(path, model, ids, local_data, remote)
    else:
        raise ValueError(
            "No valid path to dataset provided, so defaulted to building new dataset. "
            "However, no ids were provided, so dataset cannot be built."
        )

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
    ds.set_format("torch", columns=["resid"], output_all_columns=True)

    return ds


