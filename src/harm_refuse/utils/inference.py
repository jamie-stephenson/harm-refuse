"""
Utilities for model inference.
"""
from nnsight import LanguageModel
import torch
from torch import Tensor

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
        yield from _micro_batch({k: v[:m]  for k, v in b.items()}, 2*thr)
        yield from _micro_batch({k: v[m:]  for k, v in b.items()}, 2*thr)

def run_inference(
    prompts: list[str],
    model: LanguageModel,
    ids: list[int],
    batch_size: int,
    remote: bool,
) -> tuple[list[str], Tensor]:

    # Once the model goes into "remote mode" it becomes an `Envoy`
    # which does not expose `config` etc., so we store n_layers now.
    n_layers = model.config.num_hidden_layers
    tokens = _template_and_tokenize(prompts, model)
    n_rows = tokens["input_ids"].size(0)

    with model.session(remote=remote), torch.inference_mode():
        resid_batches = [].save()
        response_tokens = [].save()

        for i in range(0, n_rows, batch_size):
            print(f"Prompts processed: {i}")
            # batch the tokens *and* the mask
            batch = {k: v[i:i+batch_size] for k,v in tokens.items()}
            print(batch["input_ids"].shape)

            for b in _micro_batch(batch):
                print(b["input_ids"].shape)
                with model.generate(b, max_new_tokens=32):
                    resid_batches.append([model.model.layers[i].output[:, ids] for i in range(n_layers)])
                    response_tokens.extend(model.generator.output)

    resid = torch.cat([ 
        torch.stack([h.cpu() for h in batch], dim=1)
        for batch in resid_batches
    ])

    responses = model.tokenizer.batch_decode(
        response_tokens,
        skip_special_tokens=True,
    )

    return responses, resid
