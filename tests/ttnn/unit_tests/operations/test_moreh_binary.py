# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import math
import torch
import torch.nn as nn
import ttnn
import time
from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)


@pytest.mark.parametrize(
    "shape",
    [
        # [32, 32],  # single tile
        [1024, 32, 32],  # multiple tiles
    ],
)
def test_moreh_binary_add(shape, device):
    program_selector = 0
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input0 = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_input1 = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.add(torch_input0, torch_input1)

    tt_input0 = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_input0 = ttnn.to_device(tt_input0, device)
    tt_input1 = ttnn.from_torch(torch_input1, dtype=tt_dtype, layout=layout)
    tt_input1 = ttnn.to_device(tt_input1, device)

    tt_output = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    ttnn.moreh_binary(tt_input0, tt_input1, tt_output, 0, 0, program_selector)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output)
    passed = torch.allclose(tt_output, torch_output, atol=0.1, rtol=0.1)
    assert passed


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single tile
        [1024, 32, 32],  # multiple tiles
    ],
)
def test_moreh_fusion(shape, device):
    program_selector = 1
    slope0 = 0.1
    slope1 = 0.3
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input0 = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_input1 = torch.randint(-100, 100, shape, dtype=torch_dtype)

    torch_leaky0 = torch.nn.functional.leaky_relu(torch_input0, slope0)
    torch_leaky1 = torch.nn.functional.leaky_relu(torch_input1, slope1)
    torch_output = torch.add(torch_leaky0, torch_leaky1)

    tt_input0 = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_input0 = ttnn.to_device(tt_input0, device)
    tt_input1 = ttnn.from_torch(torch_input1, dtype=tt_dtype, layout=layout)
    tt_input1 = ttnn.to_device(tt_input1, device)

    tt_output = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    ttnn.moreh_binary(tt_input0, tt_input1, tt_output, slope0, slope1, program_selector)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    print(tt_output[0:4])

    assert_with_pcc(torch_output, tt_output)
    passed = torch.allclose(tt_output, torch_output, atol=0.1, rtol=0.1)
    assert passed


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single tile
    ],
)
@pytest.mark.parametrize(
    "seed",
    [0, 1, 2, 3, 4],
)
def test_moreh_reduce_h(shape, seed, device):
    import numpy as np

    torch.manual_seed(seed)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    torch.set_printoptions(threshold=10000, linewidth=1000)
    program_selector = 1
    dummy = 0
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input0 = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_input1 = torch.randint(-100, 100, shape, dtype=torch_dtype)

    tt_input0 = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_input0 = ttnn.to_device(tt_input0, device)
    tt_input1 = ttnn.from_torch(torch_input1, dtype=tt_dtype, layout=layout)
    tt_input1 = ttnn.to_device(tt_input1, device)

    tt_output = ttnn.from_torch(torch_input0, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    ttnn.moreh_binary(tt_input0, tt_input1, tt_output, dummy, dummy, program_selector)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    ans = torch.max(torch_input0, 0).values
    print(ans[0:16])
    print(tt_output[0][0:16])

    passed = torch.equal(ans, tt_output[0][:])
    assert passed
