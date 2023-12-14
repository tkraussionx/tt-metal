# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import pytest

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_model as nanogpt_model

from models.utility_functions import tt_to_torch_tensor, comp_allclose, comp_pcc


@pytest.mark.parametrize(
    "dtype",
    (tt_lib.tensor.DataType.BFLOAT16,),
)
@pytest.mark.parametrize(
    "pcc, prompt",
    ((0.98, "Hello, my dog is a little"),),
)
def test_nanogpt_model_real(device, pcc, prompt, dtype, reset_seeds):
    # Prepare input
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model_hf.eval()

    prompt = 8 * [prompt]
    tokenizer.add_special_tokens({"pad_token": "0"})
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    pt_model = model_hf
    pt_out = pt_model.forward(inputs.input_ids)

    config = model_hf.config

    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/NanoGPT/gpt2/"

    tt_model = nanogpt_model.TtGPT(config, device, tt_cache_path, dtype)

    tt_out = tt_model.forward(inputs.input_ids)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_model_real: Passed!")
    else:
        logger.warning("nanogpt_model_real: Failed!")

    assert does_pass
