from typing import Iterable, Optional, Tuple
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

# ---- 1) Load a skeleton (no weights) just to inspect structure ----
def load_skeleton(repo_id: str) -> nn.Module:
    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    with init_empty_weights():  # builds modules on the 'meta' device
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return model

# ---- 2) Pretty-print the module tree (top N levels) ----
def print_module_tree(model: nn.Module, max_depth: int = 2) -> None:
    print("[root]", model.__class__.__name__)
    for name, module in model.named_modules():
        if name == "":  # skip root, already printed
            continue
        depth = name.count(".")
        if depth <= max_depth:
            print("  " * depth + f"└─ {name}: {module.__class__.__name__}")

# ---- 3) Heuristic to find the "block stack" path (ModuleList of layers) ----
CANDIDATE_BLOCK_CONTAINERS: Tuple[str, ...] = (
    "transformer.h",          # GPT-2 family
    "model.layers",           # LLaMA/Mistral/Mixtral/Qwen2.x
    "gpt_neox.layers",        # GPT-NeoX / Pythia
    "model.decoder.layers",   # OPT/BART-like decoders
    "transformer.layers",     # some variants
    "backbone.layers",        # some Phi-style backbones
    "rwkv.blocks",            # RWKV
)

def find_block_container(model: nn.Module) -> Optional[str]:
    # Try common dotted paths first.
    for path in CANDIDATE_BLOCK_CONTAINERS:
        try:
            sub = model.get_submodule(path)  # raises if not present
            if isinstance(sub, nn.ModuleList) and len(sub) > 0:
                return path
        except Exception:
            pass
    # Fallback: first ModuleList of length>0 that looks like layers
    for name, sub in model.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) > 0:
            # quick sanity check: first block has attention-ish parts
            block0 = sub[0]
            has_attn = any(k in dict(block0.named_modules()).keys()
                           for k in ["self_attn", "attn", "attention"])
            if has_attn:
                return name
    return None

# ---- 4) Run this for a list of models ----
def inspect_models(repo_ids: Iterable[str], depth: int = 2):
    for rid in repo_ids:
        print("\n" + "=" * 80)
        print(f"{rid}")
        print("=" * 80)
        m = load_skeleton(rid)
        print_module_tree(m, max_depth=depth)
        path = find_block_container(m)
        if path:
            sub = m.get_submodule(path)
            print(f"\nLikely block stack: '{path}'  "
                  f"(len={len(sub) if hasattr(sub, '__len__') else 'N/A'})")
        else:
            print("\nCould not confidently identify a block stack.")

models = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "distilbert/distilgpt2",
]

if __name__ == "__main__":
    inspect_models(models)

