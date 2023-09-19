"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
"""
from functools import partial
import torch
import pytest
from loguru import logger
from transformers.generation.logits_process import LogitsProcessorList
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time


def post_process(logits, input_ids, logits_processor):
    next_token_logits = logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    return ids


def generate_next_id(Model, post_processor, input_ids, kv_cache=None, use_cache=None):
    outputs = Model(input_ids, past_key_values=kv_cache, use_cache=use_cache)
    return (
        post_processor(logits=outputs.logits, input_ids=input_ids),
        outputs.past_key_values,
    )


@pytest.mark.parametrize("batch_size", ([1, 32]))
def test_nanogpt_cpu_demo(batch_size):
    logits_processor = LogitsProcessorList()
    post_processor = partial(post_process, logits_processor=logits_processor)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    HF_model = GPT2LMHeadModel.from_pretrained("gpt2")

    HF_model.eval()

    # Prepare input
    prompt = ["Hello, my dog is a little"] * batch_size

    logger.info("Tokenizing inputs")
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, add_special_tokens=False
    )

    input_ids = inputs["input_ids"]
    generator = partial(generate_next_id, Model=HF_model, post_processor=post_processor)

    ids = input_ids
    with torch.no_grad():
        logger.info("Generating new ids")
        for i in range(32):
            start_ = time.time()
            logger.info(f"generating token {i}")

            ids, _ = generator(input_ids=ids)
            logger.info(f"iteration {i} duration {time.time() - start_}")

    answer = tokenizer.batch_decode(ids)

    for input_text, output_text in zip(prompt, answer):
        logger.info(f"input: {input_text}")
        logger.info(f"output: {output_text}")
