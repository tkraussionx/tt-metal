# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.llama2.tt.llama2_transformer import TtTransformer


import tt_lib
import torch

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def decode(tokenizer, tokens, prompt_tokens, max_gen_len):
    output_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if tokenizer.eos_id in toks:
            eos_idx = toks.index(tokenizer.eos_id)
            toks = toks[:eos_idx]
        output_tokens.append(toks)

    return output_tokens


def build_llama2_model(
    config,
    input: torch.Tensor,
    start_pos: int,
    min_prompt_len: int,
    total_len: int,
    input_text_mask: torch.Tensor,
    eos_reached: torch.Tensor,
    tokenizer,
    state_dict=None,
    base_address="",
    device=None,
) -> torch.Tensor:
    # tt_tensor = input
    # layer_split = [[0, 8], [8, 16], [16, 24], [24, config.n_layers]]
    layer_split = [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, config.n_layers]]

    prev_pos = 0
    temperature = 0.6
    top_p = 0.9

    for cur_pos in range(min_prompt_len, total_len):
        for layers in layer_split:
            if layers[0] == 0:
                freqs_cis = None
                mask = None
                model = TtTransformer(config, layers[0], layers[1], state_dict, base_address, device)
                tt_tensor, freqs_cis, mask = model(
                    input[:, prev_pos:cur_pos], prev_pos, layers[0], layers[1], freqs_cis, mask
                )
            elif layers[1] == config.n_layers:
                model = TtTransformer(config, layers[0], layers[1], state_dict, base_address, device)
                logits, freqs_cis, mask = model(tt_tensor, prev_pos, layers[0], layers[1], freqs_cis, mask)
            else:
                model = TtTransformer(config, layers[0], layers[1], state_dict, base_address, device)
                tt_tensor, freqs_cis, mask = model(tt_tensor, prev_pos, layers[0], layers[1], freqs_cis, mask)

        logits = tt_to_torch_tensor(logits).squeeze(0)

        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        next_token = next_token.reshape(-1)

        next_token = torch.where(input_text_mask[:, cur_pos], input[:, cur_pos], next_token)
        input[:, cur_pos] = next_token

        eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
        prev_pos = cur_pos
        if all(eos_reached):
            break

        del model

    return input
