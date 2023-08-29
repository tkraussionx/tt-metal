from functools import partial
import pytest
import numpy as np
import torch
from loguru import logger
from transformers.generation.logits_process import LogitsProcessorList

from transformers import AutoTokenizer
import torch.nn.functional as F

from tests.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
import time
import torch.nn.functional as F

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


class Falcon():

    def __init__(self):
        logger.info("Initializing tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)

        logger.info("Initializing CausalLM Model")
        causalLM = FalconForCausalLM.from_pretrained(MODEL_VERSION, device_map="auto")
        causalLM.eval()

        logits_processor = LogitsProcessorList()
        self.post_processor = partial(post_process, logits_processor=logits_processor)

        self.generator = partial(
            generate_next_id, causalLMModel=causalLM, post_processor=self.post_processor
        )

        self.max_seq_len = 2048

    def _pad_kv_cache(self, kv_cache):

        padded_kv_cache = []
        for d in range(len(kv_cache)):
            padded_kv_cache.append([])
            for i in range(len(kv_cache[0])):
                padded_item = F.pad(kv_cache[d][i], pad=(0, 0, 0, self.max_seq_len - kv_cache[d][i].shape[2]), mode='constant', value=-200)
                padded_kv_cache[d].append(padded_item)

        return padded_kv_cache


    def _unpad_kv_cache(self, kv_cache, kv_cache_len):
        unpadded_kv_cache = []
        for d in range(len(kv_cache)):
            unpadded_kv_cache.append([])
            for i in range(len(kv_cache[0])):
                unpadded_item = kv_cache[d][i][:, :, :kv_cache_len, :]
                unpadded_kv_cache[d].append(unpadded_item)

        return unpadded_kv_cache


    def _pad_tensor(self, T: torch.tensor, batch_size: int=32) -> torch.tensor:
        assert T.shape[0] == 1
        return torch.repeat_interleave(T, batch_size, dim=0)


    def _expand_kv_cache(self, prefill_cache, batch_size:int=32):
        decoder_size, key_len = len(prefill_cache), len(prefill_cache[0])
        padded_kv_cache = []
        for d in range(decoder_size):
            padded_kv_cache.append([])
            for i in [0, 1]:

                _x = torch.repeat_interleave(prefill_cache[d][i], batch_size, dim=0)
                padded_kv_cache[-1].append(_x)

        return padded_kv_cache

    def _prefill(self, input_ids):
        ids, kv_cache = self.generator(input_ids=input_ids, use_cache=True)
        kv_cache_len = kv_cache[0][0].shape[2]
        kv_cache = self._pad_kv_cache(kv_cache)

        return ids, kv_cache, kv_cache_len


    def _decode_stage(self, ids, kv_cache, generator, kv_cache_len, iter: int = 10):
        kv_cache = self._unpad_kv_cache(kv_cache, kv_cache_len)
        generated_ids = ids
        ids = ids[:, -1].unsqueeze(1)

        for i in range(iter):
            start_ = time.time()

            ids, kv_cache = generator(input_ids=ids, kv_cache=kv_cache, use_cache=True)

            ids = ids[:, -1].unsqueeze(1)
            generated_ids = torch.concat((generated_ids, ids), dim=1)
            logger.info(f"token {i} generated in {time.time() - start_} secs")

        return generated_ids

    def _decode_ids(self, generated_ids):
        generated_ids = generated_ids.tolist()
        text = self.tokenizer.batch_decode(generated_ids)
        logger.info(f"generated: {text}")
        return text

    def __call__(self, prompt: str, seq_len: int=15):
        self.kv_cache_len = None
        text = self.forward(prompt, seq_len)
        return text

    def forward(self, prompt:str, seq_len: int):
        tokenized_inputs = self.tokenizer(prompt, padding=False, add_special_tokens=False, return_tensors="pt")

        input_ids = tokenized_inputs["input_ids"]
        ids, kv_cache, kv_cache_len = self._prefill(input_ids)

        ids = self._pad_tensor(ids)
        kv_cache = self._expand_kv_cache(kv_cache)
        generated_ids = self._decode_stage(ids, kv_cache, self.generator, iter=seq_len-1, kv_cache_len=kv_cache_len)
        return self._decode_ids(generated_ids)



# def test_cpu_demo():
#     batch_size = 1
#     falcon = Falcon()

#     prompt_text = ["Write a poem about Valencia"] * batch_size

#     text = falcon(prompt_text)
#     logger.info(text)
#     logger.info(prompt_text)

#     for input_text, output_text in zip(prompt_text, text):
#         logger.info(f"input: {input_text}")
#         logger.info(f"output: {output_text}")
