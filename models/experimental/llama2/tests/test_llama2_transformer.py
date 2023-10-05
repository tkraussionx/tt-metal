# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
import tt_lib
from loguru import logger

from transformers import AutoModelForCausalLM
from models.experimental.llama2.reference.generation import Llama

from models.experimental.llama2.tt.llama2_configuration import ModelArgs
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose,
)
from models.experimental.llama2.tt.llama2_generation import build_llama2_model, sample_top_p, decode


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_llama2_transformer(device, pcc, model_location_generator, reset_seeds):
    llama2_path = str(model_location_generator("llama-2-7b", model_subdir="llama-2"))
    facebook_research_reference_model = Llama.build(llama2_path, llama2_path, 50, 1)
    torch_model = facebook_research_reference_model.model
    state_dict = facebook_research_reference_model.model.state_dict()
    tokenizer = facebook_research_reference_model.tokenizer

    prompt = ["A man is sitting on a roof."]

    prompt_tokens = [tokenizer.encode(s=x, bos=True, eos=False) for x in prompt]
    bsz = len(prompt_tokens)
    min_prompt_len = min(len(t) for t in prompt_tokens)

    max_gen_len = 39
    total_len = 15
    temperature = 0.6
    top_p = 0.9
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)

    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)

    input_token_length = len(tokens)
    tt_tensor = tokens.clone()
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz)
    input_text_mask = tokens != pad_id

    for cur_pos in range(min_prompt_len, total_len):
        logits = torch_model(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        next_token = next_token.reshape(-1)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)

        tokens[:, cur_pos] = next_token
        eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
        prev_pos = cur_pos
        if all(eos_reached):
            break

    torch_output_sentence = decode(tokenizer, tokens, prompt_tokens, max_gen_len)
    torch_output_sentence = [tokenizer.decode(t) for t in torch_output_sentence]
    logger.info("PyTorch Predicted answer: ")
    logger.info(torch_output_sentence)

    del torch_model

    base_address = "layers"
    ModelArgs.vocab_size = 32000
    start_pos = 0
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz)
    input_text_mask = tt_tensor != pad_id
    Tt_output = build_llama2_model(
        ModelArgs,
        tt_tensor,
        start_pos,
        min_prompt_len,
        total_len,
        input_text_mask,
        eos_reached,
        tokenizer,
        state_dict,
        base_address,
        device,
    )

    tt_output_sentence = decode(tokenizer, Tt_output, prompt_tokens, max_gen_len)
    tt_output_sentence = [tokenizer.decode(t) for t in tt_output_sentence]

    does_pass, pcc_message = comp_pcc(tokens[:, input_token_length:], Tt_output[:, input_token_length:], pcc)
    logger.info(comp_allclose(tokens[:, input_token_length:], Tt_output[:, input_token_length:]))
    logger.info(pcc_message)

    logger.info("GS Predicted answer: ")
    logger.info(tt_output_sentence)

    if does_pass:
        logger.info("Llama2Transformer Passed!")

    assert does_pass, f"Llama2Transformer does not meet PCC requirement {pcc}."
