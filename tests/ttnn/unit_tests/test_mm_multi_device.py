# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
import tt_lib as ttl


def sharded(tensor, grid_shape, device):
    if type(tensor) == torch.Tensor:
        tensor = ttnn.from_torch(tensor, ttnn.float32)

    assert tensor.shape[2] % grid_shape[0] == 0
    assert tensor.shape[3] % grid_shape[1] == 0

    core_range = ttl.tensor.CoreRange(
        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(grid_shape[1] - 1, grid_shape[0] - 1)
    )
    shard_grid = ttl.tensor.CoreRangeSet({core_range})
    shard_shape = [tensor.shape[2] // grid_shape[0], tensor.shape[3] // grid_shape[1]]
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1, shard_spec
    )
    return tensor.to(ttl.tensor.Layout.TILE).to(device, mem_config)


def test_mm_multi_device(device):
    print("asdf", device)
    torch_a = torch.randn(1, 1, 32, 32)
    torch_b = torch.randn(1, 1, 32, 32)
    golden = torch_a @ torch_b

    a = sharded(torch_a, (1, 1), device)
    b = sharded(torch_a, (1, 1), device)
    out = ttnn.experimental.tensor.mm_multi_device(a, b)
