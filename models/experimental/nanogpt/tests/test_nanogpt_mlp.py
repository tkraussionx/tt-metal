# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib

from transformers import GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
from models.experimental.nanogpt.nanogpt_helper_funcs import format_tensor, unpad_from_zero


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "dtype",
    (tt_lib.tensor.DataType.BFLOAT16,),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_nanogpt_mlp(device, pcc, dtype, reset_seeds):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.mlp"

    output_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
    )

    test_in = torch.rand(1, 43, 768)
    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    tt_test_in = format_tensor(tt_test_in, tt_lib.tensor.Layout.TILE, device, output_mem_config)
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/NanoGPT/gpt2/"

    tt_mlp = nanogpt_mlp.TtMLP(base_address, config, device, tt_cache_path, dtype)

    desired_tt_out_shape = [1, 1, 43, 768]

    tt_out = tt_mlp.forward(tt_test_in)

    pt_mlp = model_hf.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in)

    tt_out_converted = unpad_from_zero(tt_out, desired_tt_out_shape).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_mlp: Passed!")
    else:
        logger.warning("nanogpt_mlp: Failed!")

    assert does_pass
