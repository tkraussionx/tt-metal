# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import tt_lib

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def run_silu_test_sharded(device, h, w, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.silu(torch_input_tensor)

    ff1_output_shape = ttnn.Shape([h, w])
    grid = ttnn.CoreGrid(y=4, x=8)
    shard = ttnn.create_sharded_memory_config(ff1_output_shape, grid, ttnn.ShardStrategy.WIDTH)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard)

    # Takes in sharded input tensor and returns sharded output tensor in-place
    output_tensor = ttnn.silu(input_tensor, memory_config=shard)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_silu_test_sharded_tt_lib(device, h, w, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.silu(torch_input_tensor)

    ff1_output_shape = ttnn.Shape([h, w])
    grid = ttnn.CoreGrid(y=4, x=8)
    shard = ttnn.create_sharded_memory_config(ff1_output_shape, grid, ttnn.ShardStrategy.WIDTH)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard)

    input_tensor = input_tensor.value

    shard_spec_32_cores_grid = tt_lib.tensor.CoreRangeSet(
        {
            tt_lib.tensor.CoreRange(
                tt_lib.tensor.CoreCoord(0, 0),
                tt_lib.tensor.CoreCoord(7, 3),
            ),
        }
    )
    mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        tt_lib.tensor.BufferType.L1,
        tt_lib.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                h,
                w // 32,  # 32 cores
            ],
            tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    output_tensor = tt_lib.tensor.silu(input_tensor, output_mem_config=mem_config)
    output_tensor = output_tensor.cpu()
    output_tensor = output_tensor.to(tt_lib.tensor.Layout.ROW_MAJOR)
    output_tensor = output_tensor.to_torch()

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [4096])
def test_silu(device, h, w):
    run_silu_test_sharded(device, h, w)
    run_silu_test_sharded_tt_lib(device, h, w)
