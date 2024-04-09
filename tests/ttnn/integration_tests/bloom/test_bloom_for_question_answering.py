# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import BloomConfig, BloomForQuestionAnswering, BloomTokenizerFast

from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import skip_for_wormhole_b0, comp_allclose_and_pcc

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize("ttnn_model", [ttnn_optimized_functional_bloom])
def test_bloom_for_question_answering(device, use_program_cache, ttnn_model, batch_size=8, max_length=384):
    torch.manual_seed(0)

    model_name = "bigscience/bloom-560m"
    config = BloomConfig.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    torch_model = BloomForQuestionAnswering.from_pretrained(model_name).eval()

    num_heads = config.n_head

    question = "Who analyzes rock samples from drill cores in the lab?"
    context = "In the laboratory, biostratigraphers analyze rock samples from outcrop and drill cores for the fossils found in them. These fossils help scientists to date the core and to understand the depositional environment in which the rock units formed. Geochronologists precisely date rocks within the stratigraphic section in order to provide better absolute bounds on the timing and rates of deposition. Magnetic stratigraphers look for signs of magnetic reversals in igneous rock units within the drill cores. Other scientists perform stable isotope studies on the rocks to gain information about past climate."
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    num_tokens = inputs.input_ids.shape[-1]
    inputs.input_ids = inputs.input_ids.expand((batch_size, num_tokens))
    inputs.attention_mask = inputs.attention_mask.expand((batch_size, num_tokens))
    # inputs.input_ids=torch.randint(0,1000,inputs.input_ids.shape)
    # inputs.attention_mask=torch.ones(inputs.attention_mask.shape)
    print("input_ids.shape", inputs.attention_mask.shape)

    torch_output = torch_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    torch_start_logits = torch_output.start_logits
    torch_end_logits = torch_output.end_logits
    print("torch_model", torch_model)

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_functional_bloom_for_question_answering",
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_model.custom_preprocessor,
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # num_tokens = input_ids.shape[-1]
    # input_ids = input_ids.expand((batch_size, num_tokens))
    # attention_mask = attention_mask.expand((batch_size, num_tokens))

    input_ids, alibi, causal_mask = ttnn_model.preprocess_inputs(
        input_ids=input_ids,
        device=device,
        num_heads=num_heads,
        attention_mask=attention_mask,
        max_length=max_length,
    )
    # Run twice to measure the time with and without the program cache
    tt_output = ttnn_model.bloom_for_question_answering(
        config,
        input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
        torch_model=torch_model,
    )

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    tt_start_logits = tt_output[:, :num_tokens, 0]
    tt_end_logits = tt_output[:, :num_tokens, 1]

    if ttnn_model == ttnn_functional_bloom:
        assert_with_pcc(torch_start_logits, tt_start_logits, 0.96677)
        assert_with_pcc(torch_end_logits, tt_end_logits, 0.95177)
    elif ttnn_model == ttnn_optimized_functional_bloom:
        print("comp_start", comp_allclose_and_pcc(torch_start_logits, tt_start_logits, pcc=0.99))
        print("comp_end", comp_allclose_and_pcc(torch_end_logits, tt_end_logits, pcc=0.99))
        assert_with_pcc(torch_start_logits, tt_start_logits, 0.94)
        assert_with_pcc(torch_end_logits, tt_end_logits, 0.94)
    else:
        raise RecursionError("Invalid ttnn_model")

    # assert torch_start_logits.argmax() == tt_start_logits.argmax()
    # assert torch_end_logits.argmax() == tt_end_logits.argmax()

    tt_start_logits_b1 = tt_output[0, :num_tokens, 0]
    tt_end_logits_b1 = tt_output[0, :num_tokens, 1]

    torch_start_logits_b1 = torch_start_logits[0, :]
    torch_end_logits_b1 = torch_end_logits[0, :]

    # assert torch_start_logits.argmax() == tt_start_logits.argmax()
    # assert torch_end_logits.argmax() == tt_end_logits.argmax()

    print(" torch_start_logits.argmax()", torch_start_logits_b1.argmax())
    print(" torch_end_logits.argmax()", torch_end_logits_b1.argmax())
    print("")
    print(" tt_start_logits.argmax()", tt_start_logits_b1.argmax())
    print(" tt_end_logits.argmax()", tt_end_logits_b1.argmax())
    print("")
    predict_answer_tokens_torch = inputs.input_ids[0, torch_start_logits_b1.argmax() : torch_end_logits_b1.argmax() + 1]
    predict_answer_tokens_ttnn = inputs.input_ids[0, tt_start_logits_b1.argmax() : tt_end_logits_b1.argmax() + 1]

    torch_answer = tokenizer.decode(predict_answer_tokens_torch, skip_special_toekns=True)
    ttnn_answer = tokenizer.decode(predict_answer_tokens_ttnn, skip_special_toekns=True)
    print("torch output:", torch_answer)
    print("ttnn answer:", ttnn_answer)
