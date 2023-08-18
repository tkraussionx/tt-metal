import torch
from loguru import logger

import pytest
import tt_lib
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import Profiler, prep_report

from models.squeezebert.tt.squeezebert import (
    squeezebert_for_question_answering,
)
from transformers import (
    SqueezeBertForQuestionAnswering as HF_SqueezeBertForQuestionAnswering,
)
from transformers import AutoTokenizer

BATCH_SIZE = 1


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            2.25,
            11.28,
        ),
    ),
)
def test_perf(use_program_cache, expected_inference_time, expected_compile_time):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "question_answering"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
    HF_model = HF_SqueezeBertForQuestionAnswering.from_pretrained(
        "squeezebert/squeezebert-uncased"
    )

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )

    inputs = tokenizer(question, context, return_tensors="pt")

    tt_attention_mask = torch_to_tt_tensor_rm(
        inputs.attention_mask, device, put_on_device=False
    )

    tt_model = squeezebert_for_question_answering(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        tt_output = tt_model(inputs.input_ids, tt_attention_mask, inputs.token_type_ids)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        tt_output = tt_model(inputs.input_ids, tt_attention_mask, inputs.token_type_ids)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        model_name="squeezebert",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )
    compile_time = first_iter_time - second_iter_time

    logger.info(f"squeezebert inference time: {second_iter_time}")
    logger.info(f"squeezebert compile time: {compile_time}")
    tt_lib.device.CloseDevice(device)
    assert second_iter_time < expected_inference_time, "squeezebert is too slow"
    assert compile_time < expected_compile_time, "squeezebert compile time is too slow"
