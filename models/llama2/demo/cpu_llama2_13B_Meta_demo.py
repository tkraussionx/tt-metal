import sys
import torch
from transformers import AutoTokenizer
from models.llama2.llama2_utils import get_logits_processor, model_location_generator
from llama import Llama
from loguru import logger
import pytest

prompt = """If the sugar intake is low but fat intake is high, does it lead to fat storage in the body?"""
num_words = 100

def prep_tokenizer():
    # we are assuming that 7B tokenizer is the same as 13B tokenizer
    llama2_tokenizer_path = str(model_location_generator("llama-2-7b", model_subdir="llama-2"))
    tokenizer = AutoTokenizer.from_pretrained(llama2_tokenizer_path)
    return tokenizer


def test_cpu_demo(prompt = prompt, num_words = num_words, model_location_generator = model_location_generator):
    # prep tokenizier
    tokenizer = prep_tokenizer()

    # set parameters =================================================================
    llama2_path = str(model_location_generator("llama-2-13b", model_subdir="llama-2")) # fix this

    # load llama pytorch model =======================================================
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(llama2_path)

    hugging_face_reference_model.eval()

    # generate real input ============================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.tokenize(prompt)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    seq_length = input_ids.shape[1]

    logger.info(f"Initial prompt: {prompt}")
    logger.debug(f"Initial prompt ids: {input_ids}")
    logger.debug(f"Initial prompt tokens: {tokens}")

    # generate Pytorch output of num_words with generate function ====================
    logits_processor = get_logits_processor(
        input_ids, hugging_face_reference_model.config
    )

    generate_ids = hugging_face_reference_model.generate(
        input_ids, logits_processor=logits_processor, max_length=seq_length + num_words
    )

    # decode output ids
    output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    tokens = tokenizer.tokenize(output)

    # print pytorch generated reponse ================================================
    logger.debug(f"CPU's generated tokens: {tokens}")
    logger.info(f"CPU's predicted Output:\n {output}")
