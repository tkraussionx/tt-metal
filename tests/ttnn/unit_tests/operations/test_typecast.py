# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull
from loguru import logger


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32/uint32/uint16 data types")
def test_typecast_output_tensor(device):
    torch.manual_seed(0)

    h = w = 32
    from_dtype = ttnn.uint32
    to_dtype = ttnn.uint8
    torch_input_tensor = torch.full([h, w], 255)
    from_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=from_dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    output_ttnn = ttnn.typecast(from_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output_ttnn = ttnn.to_torch(output_ttnn)

    logger.debug(f"From torch_input_tensor (uint32 data type) {torch_input_tensor}")
    logger.debug(f"To torch_output_ttnn (uint8 data type {torch_output_ttnn}")
    assert torch.equal(torch_input_tensor, torch_output_ttnn)
