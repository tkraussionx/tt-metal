# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0import torch

import torch
import pytest
from torch import nn
import tt_lib
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.llama.llama_utils import (
    prepare_llama_input,
    gen_position_ids,
    get_logits_processor,
)
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    Profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    prep_report,
    comp_allclose_and_pcc,
    comp_pcc,
)
from tests.models.llama.cpu_stacked_decoders import PytorchLlamaDecoderModelStacked
from models.llama.tt.llama_stacked_decoders import TtLlamaDecoderModelStacked


def call_tt_llama_forward_func(
    profiler,
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    tokenizer,
    input_embeds,
    first_decoder_start,
    num_consecutive_decoders,
):
    # Disable compile cache
    disable_persistent_kernel_cache()

    # Perf tests keys
    first_half_first_key = "first_half_first_iter"
    first_half_second_key = "first_half_second_iter"

    position_ids_padded = gen_position_ids(input_embeds)

    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    tt_inputs = torch_to_tt_tensor_rm(input_embeds, device)

    logger.debug(f"The first call of the first half started")

    profiler.start(first_half_first_key)
    tt_llama_model = TtLlamaDecoderModelStacked(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        first_decoder_start,
        num_consecutive_decoders,
    )
    first_out = tt_llama_model(x=tt_inputs, y=position_ids_padded)
    profiler.end(first_half_first_key)
    logger.debug(f"The first call of the first half ended")

    first_half_first_iter_time = profiler.get(first_half_first_key)
    logger.info(f"FirstIterTime: {first_half_first_iter_time}")
    enable_persistent_kernel_cache()

    # The second call of the first half
    logger.debug(f"The second call of the first half started")
    profiler.start(first_half_second_key)
    tt_llama_model = TtLlamaDecoderModelStacked(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        first_decoder_start,
        num_consecutive_decoders,
    )
    first_out = tt_llama_model(x=tt_inputs, y=position_ids_padded)
    profiler.end(first_half_second_key)
    logger.debug(f"The second call of the first half ended")

    # returned type from the model is tuple
    first_out = tt_to_torch_tensor(first_out)

    first_half_second_iter_time = profiler.get(first_half_second_key)
    logger.info(f"First call of the first half: {first_half_first_iter_time}")
    logger.info(f"Second call of the first half: {first_half_second_iter_time}")

    tt_lib.device.CloseDevice(device)

    return (
        first_half_first_iter_time,
        first_half_second_iter_time,
    )


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_max_position_embeddings = 2048
BATCH_SIZE = 1

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 16

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# parameters --------------------------------------------------

# prompt = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
# They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
# Mention the large language model based product mentioned in the paragraph above:"""
prompt = "I believe the meaning of life is to"


def run_perf_llama_first_half(
    first_half_expected_inference_time,
    first_half_expected_compile_time,
):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    cpu_key = "ref_key"

    comments = "llama model with first half"

    # set parameters =================================================================
    tokenizer_name = _tokenizer_name
    llama_model_name = _llama_model_name
    base_url = _base_url
    max_position_embeddings = _max_position_embeddings

    # how many decoders to use
    first_decoder_start = _first_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    decoder_stack_list = [i for i in range(num_consecutive_decoders)]

    # load llama pytorch model ================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name
    )

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate real input =====================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    is_input_padded = True
    input_ids_padded, _, position_ids_padded = prepare_llama_input(
        prompt, tokenizer, configuration, is_input_padded
    )

    # TT output: call forward() function several times ========================
    with torch.no_grad():
        # call huggingface model
        # PyTorch output =========================================================================
        embeddings = torch.nn.Embedding(
            configuration.vocab_size, configuration.hidden_size
        )
        input_embeds = embeddings(input_ids_padded)

        profiler.start(cpu_key)
        pt_llama_first_half = PytorchLlamaDecoderModelStacked(
            hugging_face_reference_model, decoder_stack_list
        )
        pt_llama_first_half.eval()
        tt_lib.device.Synchronize()
        pytorch_out = pt_llama_first_half(x=input_embeds, y=position_ids_padded)
        profiler.end(cpu_key)

        (
            first_half_first_iter_time,
            first_half_second_iter_time,
        ) = call_tt_llama_forward_func(
            profiler,
            configuration,
            state_dict,
            base_url,
            max_position_embeddings,
            tokenizer,
            input_embeds,
            first_decoder_start,
            num_consecutive_decoders,
        )

    cpu_time = profiler.get(cpu_key)

    prep_report(
        "lammaFirstHalf",
        BATCH_SIZE,
        first_half_first_iter_time,
        first_half_second_iter_time,
        comments,
        cpu_time,
    )

    first_half_compile_time = first_half_first_iter_time - first_half_second_iter_time
    logger.info(f"LlamaFirstHalf inference time: {first_half_second_iter_time}")
    logger.info(f"LlamaFirstHalf compile time: {first_half_compile_time}")

    assert (
        first_half_second_iter_time < first_half_expected_inference_time
    ), "LlamaFirstHalf is too slow"
    assert (
        first_half_compile_time < first_half_expected_compile_time
    ), "LlamaFirstHalf compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "first_half_expected_inference_time, first_half_expected_compile_time",
    (
        (
            150,
            12,
        ),
    ),
)
def test_perf_bare_metal(
    use_program_cache,
    first_half_expected_inference_time,
    first_half_expected_compile_time,
):
    run_perf_llama_first_half(
        first_half_expected_inference_time,
        first_half_expected_compile_time,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "first_half_expected_inference_time, first_half_expected_compile_time",
    ((150, 12),),
)
def test_perf_virtual_machine(
    use_program_cache,
    first_half_expected_inference_time,
    first_half_expected_compile_time,
):
    run_perf_llama_first_half(
        first_half_expected_inference_time,
        first_half_expected_compile_time,
    )
