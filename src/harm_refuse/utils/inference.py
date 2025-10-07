"""
Utilities for model inference.
"""
from nnsight import LanguageModel
import torch
from torch import Tensor

from itertools import chain

_LONG_SEQ_TOKENS_THRESHOLD = 256  # if padded seq length exceeds this, split current inner batch in half

def _template_and_tokenize(prompts: list[str], model: LanguageModel) -> dict[str, Tensor]:
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return dict(model.tokenizer.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    )) # pyright: ignore

def _micro_batch(b, thr=_LONG_SEQ_TOKENS_THRESHOLD):
    L = int(b["attention_mask"].sum(1).max().item())
    B = int(b["input_ids"].shape[0])
    if L <= thr or B == 1:
        yield {k: (v[:, -L:] if getattr(v, "ndim", 0) == 2 else v) for k, v in b.items()}
    else:
        m = B // 2
        yield from _micro_batch({k: v[:m]  for k, v in b.items()}, int(1.5*thr))
        yield from _micro_batch({k: v[m:]  for k, v in b.items()}, int(1.5*thr))

def run_inference(
    prompts: list[str],
    model: LanguageModel,
    ids: list[int],
    batch_size: int,
    remote: bool,
) -> tuple[list[str], chain]:

    # Once the model goes into "remote mode" it becomes an `Envoy`
    # which does not expose `config` etc., so we store n_layers now.
    n_layers = model.config.num_hidden_layers
    tokens = _template_and_tokenize(prompts, model)
    n_rows = tokens["input_ids"].size(0)

    # `model.session(remote=True)` does not allow use of user defined
    # functions (e.g. `_micro_batch`) so we use them now.
    micro_batches = []
    for i in range(0, n_rows, batch_size):
        batch = {k: v[i:i+batch_size] for k, v in tokens.items()}
        micro_batches.extend(list(_micro_batch(batch)))  # runs locally

    print("Number of batches:", len(micro_batches))

    with model.session(remote=remote), torch.inference_mode():
        resid_batches = list().save()

        # Each element will be a list of length L of tensors of shape (mB,S,D)
        # mB is *micro*batch size
        response_tokens = list().save()

        for i, b in enumerate(micro_batches):
            print(f"Batches processed: {i}.")

            # TODO: Stop streamer warnings, and explore nnsight.local for early download.
            with model.generate(b, max_new_tokens=32):
                resid_batches.append([model.model.layers[l].output[:, ids] for l in range(n_layers)])
                response_tokens.extend(model.generator.output)
                
    resid = chain.from_iterable( 
        torch.unbind(torch.stack(batch, dim=1).cpu())
        for batch in resid_batches
    )

    responses = model.tokenizer.batch_decode(
        response_tokens,
        skip_special_tokens=True,
    )

    return responses, resid
