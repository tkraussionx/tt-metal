# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.experimental.mamba.tt_opt.full_model import MambaTT
from models.experimental.mamba.tt_opt import model_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99
        ),
    ),
)
def test_mamba_model_inference(device, use_program_cache, pcc: float):
    device.enable_program_cache()
    torch.manual_seed(10)
    input_tensor = torch.randn((1, 1, 32, 5120), dtype=torch.bfloat16)

    torch_output = torch.repeat_interleave(input_tensor, 32, dim=3)

    tt_input = ttnn.from_torch(input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    
    # Warmup
    tt_input1 = ttnn.permute(tt_input, (0,1,3,2))
    tt_upsample_output = ttnn.upsample(tt_input1, (1,1,32,1))
    tt_upsample_output = ttnn.permute(tt_upsample_output, (0,1,3,2))

    # Runtime
    tt_input1 = ttnn.permute(tt_input, (0,1,3,2))
    tt_upsample_output = ttnn.upsample(tt_input1, (1,1,32,1))
    tt_upsample_output = ttnn.permute(tt_upsample_output, (0,1,3,2))
    
    tt_output = ttnn.to_torch(tt_upsample_output)

    print(tt_output.shape, torch_output.shape)
    logger.info(comp_allclose(torch_output, tt_output))

    does_pass, output_pcc = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Upsample output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
