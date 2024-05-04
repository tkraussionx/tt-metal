# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import numpy as np  # remove this
import tt_lib as ttl
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import ttnn

from tt_lib.utils import (
    _nearest_y,
)


def write_to_file(tensor, filename):
    tensor = tensor.detach().cpu().numpy()
    with open(filename, "w") as f:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    for l in range(tensor.shape[3]):
                        f.write(str(tensor[i][j][k][l]) + " ")
                    f.write("\n")
                f.write("\n")


@pytest.mark.parametrize(
    "input_height, input_width, num_cores, shard_grid, shard_stragey",
    (
        (2048, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (8192, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 40, (8, 5), ttnn.ShardStrategy.WIDTH),
    ),
)
@pytest.mark.parametrize("dtype", (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16))
def test_bcast(device, input_height, input_width, num_cores, shard_grid, shard_stragey, dtype):
    torch.manual_seed(0)
    input_shape = [1, 1, input_height, input_width]
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=dtype
    )
    input_2d_height = input_tensor.get_legacy_shape()[2]
    input_2d_width = input_tensor.get_legacy_shape()[3]
    if shard_stragey == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
        shard_width = math.ceil(input_2d_width / shard_grid[1])
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
        core_grid = ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
    else:
        shard_height = input_2d_height
        shard_width = math.ceil(input_2d_width / num_cores)
        shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
        core_grid = get_shard_grid_from_num_cores(num_cores, device)

    logger.debug(f"core_grid={core_grid}")
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")
    logger.debug(f"shard_height={shard_height} and shard_width={shard_width}")

    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_width, shard_height),
        core_grid=core_grid,
        strategy=shard_stragey,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    b_weights_shape = [1, 1, 1, input_width]
    # # B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape).bfloat16()
    # sequence = np.arange(input_width)
    # B_pyt = torch.tensor(sequence).bfloat16()
    # print(B_pyt)
    B_pyt = torch.rand(size=b_weights_shape).bfloat16()
    torch_ref_output = torch.add(input, B_pyt)

    B_pyt = B_pyt.reshape(b_weights_shape)
    tt_weight = ttnn.from_torch(B_pyt, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_output = ttl.tensor.bcast(
        tt_input,
        tt_weight,
        ttl.tensor.BcastOpMath.ADD,
        ttl.tensor.BcastOpDim.H,
        output_mem_config=ttnn.get_memory_config(tt_input),
    )

    # write_to_file(ttnn.to_torch(tt_output).float(), "bcast_output.txt")
    # write_to_file(torch_ref_output.float(), "bcast_ref_output.txt")
    output_tensor = ttnn.to_torch(tt_output).float()

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing
