from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from typing import List, Dict, Any

def _mk_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    # Normalize missing keys
    for r in rows:
        r.setdefault("category", "")
    return Dataset.from_list(rows)

def _choose_first_col(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of the candidate columns found. Looking for one of: {candidates}, but had: {cols}")

def load_advbench() -> Dataset:
    """
    AdvBench harmful behaviors.
    HF hub: walledai/AdvBench (train split, columns include 'prompt').  MIT license.
    """
    ds = load_dataset("walledai/AdvBench")["train"]  # columns: ['prompt', 'target'] etc.
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": ex["prompt"],
            "source": "AdvBench",
            "category": "",           # not provided
            "source_id": f"advbench_{i}",
        })
    return _mk_dataset(rows)

def load_jbb() -> Dict[str, Dataset]:
    """
    JailbreakBench behaviors: has 'harmful' and 'benign' splits.
    Use 'Goal' as the prompt; 'Category' is available.  MIT license.
    """
    data = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful_split = data["harmful"]
    benign_split  = data["benign"]
    # Candidate columns per card: Goal, Target, Behavior, Category, Source
    def to_rows(split, label):
        rows = []
        for i, ex in enumerate(split):
            prompt = ex["Goal"]  # documented on the card
            rows.append({
                "prompt": prompt,
                "source": "JBB",
                "category": ex.get("Category", ""),
                "source_id": f"jbb_{label}_{i}",
            })
        return rows
    return {
        "harmful": _mk_dataset(to_rows(harmful_split, "harmful")),
        "harmless": _mk_dataset(to_rows(benign_split, "benign")),
    }

def load_sorry_bench() -> Dataset:
    """
    SORRY-Bench unsafe instructions across 44 categories (+ mutations).
    You must accept the license on the hub and be logged in.
    We try the CSV first; if unavailable, we fall back to JSONL files.
    """
    # Try loading the consolidated CSV (as per dataset card)
    try:
        ds = load_dataset("sorry-bench/sorry-bench-202503",
                          data_files="sorry_bench_202503.csv")["train"]
    except Exception:
        # Fallback: load all question*.jsonl files (base + mutations)
        ds = load_dataset("sorry-bench/sorry-bench-202503",
                          data_files="question*.jsonl")["train"]

    cols = ds.column_names
    # Candidate text fields seen in SORRY-Bench files
    text_col = _choose_first_col(cols, ["question", "prompt", "instruction", "query"])
    # Candidate category-ish fields
    cat_col = None
    for cand in ["category", "subcategory", "safety_category", "domain", "category_id"]:
        if cand in cols:
            cat_col = cand
            break

    rows = []
    for i, ex in enumerate(ds):
        cat_val = ex.get(cat_col, "") if cat_col else ""
        rows.append({
            "prompt": ex[text_col],
            "source": "SORRY-Bench",
            "category": str(cat_val),
            "source_id": f"sorry_{i}",
        })
    return _mk_dataset(rows)

def load_xstest() -> Dict[str, Dataset]:
    """
    XSTest includes 250 safe + 200 unsafe prompts (labels 'safe'/'unsafe').
    Use 'prompt' as text; include 'type' or 'focus' in category if present.  CC-BY-4.0.
    """
    ds = load_dataset("walledai/XSTest")["test"]  # columns include: prompt, label ('safe'/'unsafe'), type, focus, note
    rows_safe, rows_unsafe = [], []
    for i, ex in enumerate(ds):
        cat_bits = []
        if ex.get("type"):  cat_bits.append(ex["type"])
        if ex.get("focus"): cat_bits.append(ex["focus"])
        category = " | ".join(cat_bits)
        row = {
            "prompt": ex["prompt"],
            "source": "XSTest",
            "category": category,
            "source_id": f"xstest_{i}",
        }
        if str(ex.get("label", "")).lower() == "safe":
            rows_safe.append(row)
        else:
            rows_unsafe.append(row)
    return {
        "harmless": _mk_dataset(rows_safe),
        "harmful": _mk_dataset(rows_unsafe),
    }

def load_alpaca() -> Dataset:
    """
    Alpaca instruction-tuning data (harmless). Use 'instruction' + optional 'input' to form a single prompt.
    Prefer the popular cleaned variant: yahma/alpaca-cleaned (CC-BY-4.0).
    """
    ds = load_dataset("yahma/alpaca-cleaned")["train"]  # columns: instruction, input, output
    rows = []
    for i, ex in enumerate(ds):
        instr = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        # Compact prompt: instruction, with input appended if present
        prompt = instr if not inp else f"{instr}\n\nInput:\n{inp}"
        rows.append({
            "prompt": prompt,
            "source": "Alpaca",
            "category": "",
            "source_id": f"alpaca_{i}",
        })
    return _mk_dataset(rows)

def build_datasetdict() -> DatasetDict:
    # Load each component
    advbench = load_advbench()
    jbb = load_jbb()
    sorrybench = load_sorry_bench()
    xstest = load_xstest()
    alpaca = load_alpaca()

    # Concatenate into two splits
    harmful = concatenate_datasets([
        advbench,
        jbb["harmful"],
        sorrybench,
        xstest["harmful"],
    ])

    harmless = concatenate_datasets([
        jbb["harmless"],
        xstest["harmless"],
        alpaca,
    ])

    return DatasetDict({"harmful": harmful, "harmless": harmless})

if __name__ == "__main__":
    big = build_datasetdict()

    # Quick sanity checks & usage examples
    print(big)
    print("Harmful count :", len(big["harmful"]))
    print("Harmless count:", len(big["harmless"]))

    # Example: index directly
    ex = big["harmful"][0]
    print("Example harmful prompt from", ex["source"], "->", ex["prompt"][:120].replace("\n", " "), "...")

    # Example: filter by source (vectorized)
    jbb_harmless = big["harmless"].filter(lambda r: r["source"] == "JBB")
    print("JBB harmless examples:", len(jbb_harmless))

    # Save locally for reuse
    big.save_to_disk("big_safety_prompts_ds")
    # Later: from datasets import load_from_disk; big = load_from_disk("big_safety_prompts_ds")
