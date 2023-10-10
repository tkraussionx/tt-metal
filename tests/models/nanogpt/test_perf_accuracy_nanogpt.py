# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
from loguru import logger
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import evaluate
import pytest
from models.utility_functions import torch_to_tt_tensor_rm
from tests.models.nanogpt.dataset_utils import get_data
from models.nanogpt.tt.nanogpt import *

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    profiler,
)
from models.utility_functions import prep_report

BATCH_SIZE = 1


def run_demo_nanogpt(
    model_name,
    expected_inference_time,
    expected_compile_time,
    iterations,
    max_new_tokens,
    temperature,
    model_location_generator,
    device,
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    bert_score = evaluate.load("bertscore")

    disable_persistent_kernel_cache()

    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "accuracy_loop"
    cpu_key = "ref_key"

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    tt_model = nanogpt_model(device)
    tt_model.eval()

    prompt = "Hello, my dog is a little"
    topk = None
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)

    with torch.no_grad():
        profiler.start(cpu_key)
        generated_output = model.generate(**inputs)
        torch_answer = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model.generate(inputs.input_ids, max_new_tokens, temperature, top_k=topk)

        tt_answer = tokenizer.decode(tt_output[0], skip_special_tokens=True)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_answer

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model.generate(inputs.input_ids, max_new_tokens, temperature, top_k=topk)

        tt_answer = tokenizer.decode(tt_output[0], skip_special_tokens=True)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_answer

        profiler.start(third_key)

        calculated_label = []
        input_loc = model_location_generator("nanogpt/inputs/hellaswag_validation.jsonl")
        val_examples = get_data(input_loc)

        for i in range(iterations):
            prompt = val_examples[i].input_sentence
            inputs = tokenizer(prompt, return_tensors="pt", padding=False)
            tt_out = tt_model.generate(inputs.input_ids, max_new_tokens, temperature, top_k=topk)

            answer = tokenizer.decode(tt_out[0], skip_special_tokens=True)
            prediction = answer[len(prompt) + 1 :]

            score = []
            for end in val_examples[i].endings:
                results = bert_score.compute(predictions=[prediction], references=[end], lang="en")
                score.append(results["f1"])

            calculated_label.append(score)

        calculated_label = np.array(calculated_label)
        golden_labels = np.array([x.label for x in val_examples])

        accuracy = np.mean(calculated_label.argmax(1) == golden_labels[:iterations])

        logger.info("Accuracy: ")
        logger.info(accuracy)

    tt_lib.device.Synchronize()
    profiler.end(third_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)

    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name=f"nanoGPT",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=f"nanoGPT",
        inference_time_cpu=cpu_time,
    )

    logger.info(f"{model_name} inference time: {second_iter_time}")
    logger.info(f"{model_name} compile time: {compile_time}")
    logger.info(f"{model_name} inference for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration, max_new_tokens, temperature",
    (("gpt2", 60, 60, 10, 25, 0.8),),
)
def test_demo_bare_metal_nanogpt(
    model_name,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    iteration,
    max_new_tokens,
    temperature,
    model_location_generator,
    device,
):
    run_demo_nanogpt(
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        max_new_tokens,
        temperature,
        model_location_generator,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "model_name,expected_inference_time, expected_compile_time, iteration, max_new_tokens, temperature",
    (("gpt2", 60, 60, 10, 25, 0.8),),
)
def test_demo_virtual_machine_nanogpt(
    model_name,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    iteration,
    max_new_tokens,
    temperature,
    model_location_generator,
    device,
):
    run_demo_nanogpt(
        model_name,
        expected_inference_time,
        expected_compile_time,
        iteration,
        max_new_tokens,
        temperature,
        model_location_generator,
        device,
    )
