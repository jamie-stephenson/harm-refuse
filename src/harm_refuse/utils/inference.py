"""
Utilities for model inference.
"""
from nnsight import LanguageModel
import torch
from torch import Tensor
from tqdm.auto import tqdm


def _template_and_tokenize(prompts: list[str], model: LanguageModel) -> Tensor:
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return model.tokenizer.apply_chat_template(
        chats,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ) # pyright: ignore[reportReturnType]

def run_inference(
    prompts: list[str],
    model: LanguageModel,
    ids: list[int],
    batch_size: int,
) -> tuple[list[str], list[Tensor]]:
    layers = model.model.layers

    # Once the model goes into "remote mode" it becomes an `Envoy`
    # which does not expose `config` etc., so we store n_layers now.
    n_layers = model.config.num_hidden_layers
    
    with model.session(remote=True) and torch.inference_mode():
        resid_list = []
        responses_tokens = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]
            tokens = _template_and_tokenize(batch, model)

            with model.generate(tokens, max_new_tokens=256):
                # Save the residual stream after *one* forward pass
                resid_list_batch = [layers[i].output[:, ids] for i in range(n_layers)].save()

                # Save output tokens after *all* forwards passes
                response_tokens = model.generator.output.save()

            resid_batch = torch.stack(resid_list_batch, dim=1).cpu()
            resid_list.extend(resid_batch)
            responses_tokens.extend(response_tokens)

        responses = model.tokenizer.batch_decode(
            responses_tokens,
            skip_special_tokens=True,
        )

    return responses, resid_list
