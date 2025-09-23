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
import json

# unified features for pre-inference prompt pool
_PROMPT_FEATS = Features({
    "prompt": Value("string"),
    "source": Value("string"),
    "is_harmful": Value("bool"),
})

# outer batching for calls to run_inference (inner batching lives inside run_inference)
_OUTER_BATCH_SIZE = 1024

# --------- helpers to load prompts only (no inference here) ---------

def _ds_from_cols(prompts: list[str], source: str, is_harmful: bool) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "source": [source] * len(prompts),
            "is_harmful": [is_harmful] * len(prompts),
        },
        features=_PROMPT_FEATS,
    )

def _advbench_prompts() -> Dataset:
    ds = load_dataset("walledai/AdvBench", split="train")
    return _ds_from_cols(list(ds["prompt"]), "AdvBench", True)

def _jbb_prompts() -> Dataset:
    data = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = _ds_from_cols(list(data["harmful"]["Goal"]), "JBB", True)
    benign  = _ds_from_cols(list(data["benign"]["Goal"]),  "JBB", False)
    return concatenate_datasets([harmful, benign])

def _sorry_bench_prompts() -> Dataset:
    ds = load_dataset(
        "sorry-bench/sorry-bench-202503",
        token="",
        split="train[:1000]",
    ).map(lambda e: {"turns": e["turns"][0]})
    return _ds_from_cols(list(ds["turns"]), "SORRY-Bench", True)

def _xstest_prompts() -> Dataset:
    ds = load_dataset("walledai/XSTest")["test"]
    safe   = ds.filter(lambda x: x["label"] == "safe")
    unsafe = ds.filter(lambda x: x["label"] == "unsafe")
    safe_ds   = _ds_from_cols(list(safe["prompt"]), "XSTest", False)
    unsafe_ds = _ds_from_cols(list(unsafe["prompt"]), "XSTest", True)
    return concatenate_datasets([safe_ds, unsafe_ds])

def _alpaca_prompts() -> Dataset:
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
    ds = ds.map(
        lambda e: {"prompt": e["instruction"] + (("\n" + e["input"]) if e.get("input") else "")},
        remove_columns=["instruction", "input"]
    )
    return _ds_from_cols(list(ds["prompt"]), "Alpaca", False)

def _collect_prompts_from_local(data_dir: Path) -> Dataset:
    """Load newline-delimited JSON prompts from local data and return dataset with prompt/source/is_harmful only. Here, 'local data' refers to the data in the orginal paper's repo."""

    harm_overrides = {
        "alpaca_data_instruction.json": False,
        "example_test_harmless.json": False,
        "xstest-harmless.json": False,
    }

    prompt_datasets = []
    large_files = ["Fuzzer", "example_test", "human-seed"]
    for path in sorted(data_dir.glob("*.json")):
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
                records.append(prompt)
        if not records:
            continue

        is_harmful = harm_overrides.get(path.name)
        if is_harmful is None:
            lowered = path.stem.lower()
            is_harmful = not ("harmless" in lowered or "alpaca" in lowered)

        prompt_datasets.append(_ds_from_cols(records, path.stem, is_harmful))

    if not prompt_datasets:
        raise ValueError(f"No JSON prompts found in {data_dir}")

    if len(prompt_datasets) == 1:
        return prompt_datasets[0]
    return concatenate_datasets(prompt_datasets)

# --------- build unified prompt pool (no inference yet) ---------

def _build_prompt_pool(
    use_local_data: bool,
) -> Dataset:
    if use_local_data:
        return _collect_prompts_from_local(Path("./local_data/"))

    components = [
        _advbench_prompts,
        _jbb_prompts,
        _sorry_bench_prompts,
        _xstest_prompts,
        _alpaca_prompts,
    ]
    return concatenate_datasets([fn() for fn in components])

# --------- attach model outputs to unified dataset ---------

def _attach_outputs(
    prompt_ds: Dataset,
    model: LanguageModel,
    ids: list[int],
    remote: bool,
    inner_batch_size: int = 128,
) -> Dataset:
    """Run outer-batched inference over prompts, attach is_refused and resid per row."""
    prompts = list(prompt_ds["prompt"])
    N = len(prompts)

    # resid geometry
    L = model.config.num_hidden_layers
    S = len(ids)
    D = model.config.hidden_size

    all_is_refused: list[bool] = []
    all_resid_arrays: list[np.ndarray] = []

    for start in range(0, N, _OUTER_BATCH_SIZE):
        batch_prompts = prompts[start:start + _OUTER_BATCH_SIZE]
        responses, resid = run_inference(
            prompts=batch_prompts,
            model=model,
            ids=ids,
            batch_size=inner_batch_size,
            remote=remote,
        )  # resid expected shape: [B, L, S, D] after stacking

        # ensure CPU numpy and correct dtype
        resid_np = resid.detach().cpu().numpy()

        all_is_refused.extend([is_refused(r) for r in responses])
        all_resid_arrays.extend([resid_np[i] for i in range(resid_np.shape[0])])

        print("Prompts processed: ", start)

    # create new dataset with appended columns in original order
    feats = Features({
        "prompt": Value("string"),
        "source": Value("string"),
        "is_harmful": Value("bool"),
        "is_refused": Value("bool"),
        "resid": Array3D((L, S, D), "float32"),
    })

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "source": list(prompt_ds["source"]),
            "is_harmful": list(prompt_ds["is_harmful"]),
            "is_refused": all_is_refused,
            "resid": all_resid_arrays,
        },
        features=feats,
    )

# --------- pipeline entrypoints (rest of file) ---------

def _build_base_dataset(
    path: Path,
    model: LanguageModel,
    ids: list[int],
    use_local_data: bool,
    remote: bool,
) -> Dataset:
    prompt_pool = _build_prompt_pool(use_local_data)

    # run outer-batched inference and attach outputs
    base = _attach_outputs(
        prompt_ds=prompt_pool,
        model=model,
        ids=ids,
        remote=remote,
    )

    base.save_to_disk(path)
    return base

def get_dataset(
    path: Path,
    model: LanguageModel,
    n_samples: int,
    use_local_data: bool,
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
    """
    if path.exists():
        ds = load_from_disk(path)
    elif ids:
        ds = _build_base_dataset(path, model, ids, use_local_data, remote)
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
            raise ValueError(f"Not enough examples in split '{k}' ({m.sum()}<{n_samples}).")
        else:
            splits[k] = ds.select(
                rng.choice(np.where(m)[0], size=n_samples, replace=False).tolist()
            ) 

    ds = DatasetDict(splits)
    ds.set_format("torch", columns=["resid"], output_all_columns=True)
    return ds
