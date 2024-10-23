# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (128, 128)])
def test_typecast_bf16_to_bfp8_b(device, shape):
    torch.manual_seed(2023)
    npu_dtype = ttnn.bfloat8_b
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    # Range [1, 2) to make sure all the numbers share the same exponent
    torch_input = torch.rand(shape, dtype=cpu_dtype, requires_grad=True) + 1

    tt_input = ttnn.from_torch(torch_input, dtype=npu_dtype, layout=npu_layout, device=device)
    tt_input_cpu = ttnn.to_torch(tt_input)

    passed = torch.equal(tt_input_cpu, torch_input)

    print(tt_input_cpu[0, 0:32])
    print(torch_input[0, 0:32])

    assert passed
