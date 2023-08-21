from functools import partial
import numpy as np
import torch
from loguru import logger
from transformers.generation.logits_process import LogitsProcessorList

model_version = "tiiuae/falcon-rw-1b"
from transformers import AutoTokenizer
import torch.nn.functional as F
from transformers.models.falcon import *
import time


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


tokenizer = AutoTokenizer.from_pretrained(model_version)

BATCH = 1
prompt_text = ["Write a poem about Valencia"] * BATCH
tokenized_inputs = tokenizer(
    prompt_text, padding=False, add_special_tokens=False, return_tensors="pt"
)

input_ids = tokenized_inputs["input_ids"]


causalLM = FalconForCausalLM.from_pretrained(model_version, device_map="auto")
causalLM.eval()

post_process = partial(post_process, logits_processor=LogitsProcessorList())

generator = partial(
    generate_next_id,
    causalLMModel=causalLM,
    post_processor=post_process,
    input_ids=input_ids,
)


ids = input_ids
generated_ids = ids.tolist()[0]
ids, kv_cache = generator(input_ids=ids)

generated_ids.append(ids[0][-1].item())
start_ = time.time()
kv_list = []

for i in range(32):
    start = time.time()

    ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)

    generated_ids.append(ids[0][-1].item())

    text = tokenizer.decode(ids[0])
    ids = ids[:, -1].unsqueeze(1)
    end = time.time()

    kv_list.append(ids)
    print(f"at iteration {i}, duration: {end - start}")
cached_tensor = torch.cat(kv_list, dim=0)

ids, kv_cache = generator(input_ids=cached_tensor)
generated_ids.append(ids[0][-1].item())

for i in range(68):
    ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)

    generated_ids.append(ids[0][-1].item())

    text = tokenizer.decode(ids[0])
    ids = ids[:, -1].unsqueeze(1)
    end = time.time()
    print(f"at iteration {i+32}, duration: {end - start}")
print(f"total duration: {time.time() - start_}")
text = tokenizer.decode(generated_ids)
logger.info(text)
