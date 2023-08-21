from functools import partial
import numpy as np
import torch
from loguru import logger

from transformers.generation.logits_process import LogitsProcessorList

from tests.python_api_testing.models.falcon.falcon_common import MODEL_VERSION

from tests.python_api_testing.models.falcon.reference.hf_falcon_model import (
    RWForCausalLM,
)

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

def post_process(logits, input_ids, logits_processor):
    next_token_logits = logits[:, -1, :]

    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    return ids

def generate_next_id(causalLMModel, post_processor, input_ids, kv_cache=None, use_cache=None):
    outputs = causalLMModel(input_ids, past_key_values=kv_cache, use_cache=use_cache)
    return post_processor(logits=outputs.logits, input_ids=input_ids), outputs.past_key_values


def test_cpu_demo_no_kv():

    logits_processor = LogitsProcessorList()
    post_processor = partial(post_process, logits_processor=logits_processor)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
    prompt_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
    prompt_text = "Write a poem about Valencia"
    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized_inputs['input_ids']

    logger.info("Initializing CausalLM Model")
    causalLM = RWForCausalLM.from_pretrained(MODEL_VERSION)
    causalLM.eval()

    generator = partial(generate_next_id, causalLMModel=causalLM, post_processor=post_processor)

    import time
    start = time.time()

    logger.info("Generating new ids")
    ids = input_ids
    for i in range(40):
        logger.info(f"generating token {i}")
        ids, kv = generator(input_ids=ids)

    end = time.time()
    logger.info(f"duration: {end - start}")

    logger.info(f"Input Prompt: {prompt_text}")

    logger.info("decoding to text")
    text = tokenizer.decode(ids[0])
    logger.info("Total output (including input): ")
    logger.info(text)

'''
50 tokens
input prompt: Write a poem about Valencia
Valencia, the city of the sun,
A place of beauty, of fun,
A place of culture, of art,
Where the people are warm, and the heart.
The city of the sun, where the sky
'''

def test_cpu_demo_KV():

    logits_processor = LogitsProcessorList()
    post_processor = partial(post_process, logits_processor=logits_processor)

    logger.info("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)

    # prompt_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
    prompt_text = "Write a poem about Valencia"
    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")

    input_ids = tokenized_inputs['input_ids']
    logger.info(f"input_ids shape {input_ids.shape}")
    ids = input_ids


    logger.info("Initializing CausalLM Model")
    causalLM = RWForCausalLM.from_pretrained(MODEL_VERSION)
    causalLM.eval()

    generator = partial(generate_next_id, causalLMModel=causalLM, post_processor=post_processor, input_ids=input_ids)

    logger.info(f"input_ids shape {input_ids.shape}")
    ids = input_ids
    generated_ids = ids.tolist()[0]

    logger.info("Generating new ids")

    import time
    start_ = time.time()

    ids, kv_cache = generator(input_ids=ids)
    generated_ids.append(ids[0][-1].item())

    for i in range(50):
        start = time.time()

        ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)
        generated_ids.append(ids[0][-1].item())

        logger.info(generated_ids)

        text = tokenizer.decode(ids[0])
        ids = ids[:, -1].unsqueeze(1)
        end = time.time()

        logger.info(f"at iteration {i}, duration: {end - start}")


    text = tokenizer.decode(generated_ids)
    logger.info(text)

    logger.info(f"total duration: {time.time() - start_}")
