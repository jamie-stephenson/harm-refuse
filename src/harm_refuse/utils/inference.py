"""
Utilities for model inference.
"""
from nnsight import LanguageModel
import torch
from torch import Tensor


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

def run_inference(
    prompts: list[str],
    model: LanguageModel,
    ids: list[int],
    batch_size: int,
) -> tuple[list[str], Tensor]:

    # Once the model goes into "remote mode" it becomes an `Envoy`
    # which does not expose `config` etc., so we store n_layers now.
    n_layers = model.config.num_hidden_layers
    tokens = _template_and_tokenize(prompts, model)
    n_rows = tokens["input_ids"].size(0)

    with model.session(remote=True), torch.inference_mode():
        resid_batches = [].save()
        response_tokens = [].save()

        for i in range(0, n_rows, batch_size):
            print(f"Prompts processed: {i}")
            # batch the tokens *and* the mask
            batch = {k: v[i:i+batch_size] for k,v in tokens.items()}

            with model.generate(batch, max_new_tokens=64):
                # Save the residual stream after *one* forward pass
                resid_batches.append([model.model.layers[i].output[:, ids] for i in range(n_layers)])

                # Save output tokens after *all* forwards passes
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
