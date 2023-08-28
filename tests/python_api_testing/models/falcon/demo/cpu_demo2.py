from functools import partial
import pytest
import numpy as np
import torch
from loguru import logger
from transformers.generation.logits_process import LogitsProcessorList

from transformers import AutoTokenizer
import torch.nn.functional as F

from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
import time

# MODEL_VERSION = falcon1b
MODEL_VERSION = "tiiuae/falcon-7b-instruct"



def post_process(logits, input_ids, logits_processor):
    next_token_logits = logits[:, -1, :]
    next_tokens_scores = logits_processor(input_ids, next_token_logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    return ids


def generate_next_id(
    causalLMModel, post_processor, input_ids, kv_cache=None, use_cache=None
):
    outputs = causalLMModel(input_ids, past_key_values=kv_cache, use_cache=use_cache)
    return (
        post_processor(logits=outputs.logits, input_ids=input_ids),
        outputs.past_key_values,
    )


def prefill(input_ids, generator):
    ids, kv_cache = generator(input_ids=input_ids, use_cache=True)
    return ids, kv_cache


def decode_stage(ids, kv_cache, generator, iter: int = 15):

    generated_ids = ids
    ids = ids[:, -1].unsqueeze(1)

    for i in range(iter):
        start_ = time.time()

        ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)

        ids = ids[:, -1].unsqueeze(1)
        generated_ids = torch.concat((generated_ids, ids), dim=1)
        logger.info(f"token {i} generated in {time.time() - start_} secs")

    return generated_ids


def setup():
    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)

    logger.info("Initializing CausalLM Model")
    causalLM = FalconForCausalLM.from_pretrained(MODEL_VERSION, device_map="auto")
    causalLM.eval()

    logits_processor = LogitsProcessorList()
    post_processor = partial(post_process, logits_processor=logits_processor)

    generator = partial(
        generate_next_id, causalLMModel=causalLM, post_processor=post_processor
    )

    return tokenizer, generator, post_processor


def process_kv_cache(kv_cache):
    return kv_cache


def pad_kv_cache(prefill_cache, batch_size:int=32):
    decoder_size, key_len = len(prefill_cache), len(prefill_cache[0])
    logger.info(f"decoder size, key len: {decoder_size, key_len}")
    padded_kv_cache = []
    for d in range(decoder_size):
        padded_kv_cache.append([])
        logger.info(f"{d}: {len(prefill_cache[d])}")
        for i in [0, 1]:  # key, query

            logger.info(type(prefill_cache[d][i]))
            logger.info(len(prefill_cache[d][i]))
            logger.info(prefill_cache[d][i].shape)
            _x = torch.repeat_interleave(prefill_cache[d][i], batch_size, dim=0)
            padded_kv_cache[-1].append(_x)

    return padded_kv_cache


def pad_tensor(T: torch.tensor, batch_size: int=32)-> torch.tensor:
    assert T.shape[0] == 1
    return torch.repeat_interleave(T, batch_size, dim=0)


def test_cpu_demo():
    batch_size = 1
    # setup
    tokenizer, generator, post_processor = setup()

    prompt_text = ["Write a poem about Valencia"] * batch_size

    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(
        prompt_text, padding=False, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = tokenized_inputs["input_ids"]

    logger.info("Generating new ids")
    ids = input_ids

    #prefill
    ids, kv_cache = prefill(input_ids, generator)

    # pad kv_cache to batch_size 32
    ids = pad_tensor(ids)
    kv_cache = pad_kv_cache(kv_cache)
    # decode Fix number of times
    generated_ids = decode_stage(ids, kv_cache, generator)

    logger.info(generated_ids.shape)

    generated_ids = generated_ids.tolist()
    text = tokenizer.batch_decode(generated_ids)



    for input_text, output_text in zip(prompt_text, text):
        logger.info(f"input: {input_text}")
        logger.info(f"output: {output_text}")
