# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import pytest
import tt_lib
from models.utility_functions import profiler
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import prep_report

from transformers import BloomForCausalLM, BloomTokenizerFast

from loguru import logger
import pytest

from models.bloom.tt.bloom_model import TtBloomModel


BATCH_SIZE = 1


def run_perf_bloom(expected_inference_time, expected_compile_time, device):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    model_name = "bigscience/bloom-560m"
    tokenizer_name = "bigscience/bloom-560m"
    comments = "560M"

    HF_model_top = BloomForCausalLM.from_pretrained(model_name, torchscript=False)
    HF_model_top.eval()

    config = HF_model_top.config

    state_dict = HF_model_top.state_dict()
    base_address = "transformer"

    tt_model = TtBloomModel(config, state_dict, base_address, device)
    HF_model = HF_model_top.transformer

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_name)
    inputs = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."

    tokenized = tokenizer(inputs, return_tensors="pt")
    input_ids = tokenized.input_ids

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(input_ids)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(input_ids)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(input_ids)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name="bloom",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"bloom {comments} inference time: {second_iter_time}")
    logger.info(f"bloom {comments} compile time: {compile_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            1.80,
            16,
        ),
    ),
)
def test_perf_bare_metal(
    use_program_cache, expected_inference_time, expected_compile_time, device
):
    run_perf_bloom(expected_inference_time, expected_compile_time, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            2.05,
            20,
        ),
    ),
)
def test_perf_virtual_machine(
    use_program_cache, expected_inference_time, expected_compile_time, device
):
    run_perf_bloom(expected_inference_time, expected_compile_time, device)
