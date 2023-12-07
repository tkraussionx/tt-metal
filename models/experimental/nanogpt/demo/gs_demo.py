# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import pytest

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_model as nanogpt_model
from models.experimental.nanogpt.nanogpt_utils import generate
from models.utility_functions import tt_to_torch_tensor, comp_allclose, comp_pcc


@pytest.mark.parametrize(
    "batch_size",
    ((2),),
)
def test_gs_demo(batch_size, device):
    prompts = batch_size * [
        "Hello, my dog is a little",
    ]

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model_hf.eval()

    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    config = model_hf.config

    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/NanoGPT/gpt2/"
    dtype = tt_lib.tensor.DataType.BFLOAT16

    tt_model = nanogpt_model.TtGPT(config, device, tt_cache_path, dtype)

    tt_output = generate(
        idx=inputs.input_ids,
        tt_model=tt_model,
        config=config,
        tokenizer=tokenizer,
        max_new_tokens=30,
        device=device,
    )

    logger.info("Input Prompt")
    logger.info(prompts)
    logger.info("nanoGPT Model output")
    logger.info(tt_output)
