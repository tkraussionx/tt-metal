# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
import torch
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast

from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


def get_expected_times(functional_bloom):
    return {
        ttnn_functional_bloom: (15.0, 9.2),
        ttnn_optimized_functional_bloom: (12, 0.85),
    }[functional_bloom]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("functional_bloom", [ttnn_optimized_functional_bloom])
def test_performance_of_causal_lm(device, use_program_cache, functional_bloom, max_length=128, batch_size=8):
    disable_persistent_kernel_cache()

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    num_heads = config.n_head

    # context = "Two women in a child are shown in a canoe while a man pulls the canoe while standing in the water, with other individuals visible in the background."
    # context = "A cartoon animation video is shown with people wandering around and rockets being shot."
    # context = "Then he takes a piece of bark and rubs the powered stone pieces onto it. The stone particles stick to the wet piece of wood."
    # context = "The family enjoys eating the desert together. The people in the restaurant laugh at the man and he wonders what they are doing."
    # context = "She gets them some water to gargle in their mouths. The boy and girl begin playing in the sink."
    # context = "A sea is shown with a green forest on seashore. Blond man is standing in seashore and talking to the camera and surfing big waves on the sea."
    # context = "A camera pans away from a road and shows a person's feet moving around. Several shots are shown of people tying their shoes, hitting a button, and checking a watch."
    # context = "She turns a dial on the appliance and sets it back down. She picks the knife back up and places it on the appliance."
    context = "Eight people are standing at the studio, one woman walked at the back of the group."
    # context = "A man is seen speaking to the camera while a shot of a person playing basketball shows behind."

    inputs = tokenizer.encode_plus(context, return_tensors="pt")

    if functional_bloom == ttnn_functional_bloom:
        tt_model_name = f"ttnn_{model_name}"
    elif functional_bloom == ttnn_optimized_functional_bloom:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown functional_bloom: {functional_bloom}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: BloomForCausalLM.from_pretrained(model_name).eval(),
        device=device,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_tokens = input_ids.shape[-1]
    input_ids = input_ids.expand((batch_size, num_tokens))
    attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = functional_bloom.preprocess_inputs(
        input_ids=input_ids, device=device, num_heads=num_heads, attention_mask=attention_mask, max_length=max_length
    )

    # TODO: don't modify the config globally. Pass it into the functions instead
    ttnn_optimized_functional_bloom.ASSUME_FUSED_SOFTMAX = False

    durations = []
    for _ in range(2):
        start = time.time()
        tt_output = functional_bloom.bloom_for_causal_lm(config, input_ids, alibi, causal_mask, parameters=parameters)
        # tt_output = ttnn.from_device(tt_output)
        tt_output = ttnn.from_torch(tt_output, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        end = time.time()

        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times(functional_bloom)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Tokens per second: {1 / inference_time * batch_size}")
