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
    "which_relu, bound",
    [
        [0, 0],  # vanilla relu
        [1, 3],  # relu_max
        [2, 3],  # relu_min
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single tile
        [1024, 32, 32],  # multiple tiles
    ],
)
def test_moreh_relu(which_relu, bound, shape, device):
    ttnn.enable_program_cache(device)
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    if which_relu == 0:
        torch_output = torch.relu(torch_input)
    elif which_relu == 1:
        torch_output = torch.relu(torch.clamp(torch_input, min=None, max=bound))
    else:
        torch_output = torch.relu(torch.clamp(torch_input, min=bound, max=None))

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    print(f"tt_input\n{tt_input}")
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)
    print(f"tt_output\n{tt_output}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed


def test_moreh_relu_program_cache(device):
    ttnn.enable_program_cache(device)
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    # Running the first time will compile the program and cache it
    shape = [512, 32, 32]
    which_relu = 1
    bound = 3

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.relu(torch.clamp(torch_input, min=None, max=bound))

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    start_time = time.time()
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    print(f"duration of the 1st run: {duration_ms}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed

    # Running the subsequent time will use the cached program
    shape = [512, 32, 32]
    which_relu = 1
    bound = 10

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.relu(torch.clamp(torch_input, min=None, max=bound))

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    start_time = time.time()
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    print(f"duration of the 2nd run: {duration_ms}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed

    # Running the subsequent time will use the cached program
    shape = [1024, 32, 32]
    which_relu = 1
    bound = 3

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.relu(torch.clamp(torch_input, min=None, max=bound))

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    start_time = time.time()
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    print(f"duration of the 3rd run: {duration_ms}")

    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed


@pytest.mark.parametrize(
    "which_relu, slope",
    [
        [3, 0.1],  # relu_min
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single tile
        [1024, 32, 32],  # multiple tiles
    ],
)
def test_moreh_leaky_ans(which_relu, slope, shape, device):
    ttnn.enable_program_cache(device)
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.nn.functional.leaky_relu(torch_input, slope)

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    print(f"tt_input\n{tt_input}")
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=slope)
    print(f"tt_output\n{tt_output}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output)
    passed = torch.allclose(tt_output, torch_output, atol=0.1, rtol=0.1)
    assert passed
