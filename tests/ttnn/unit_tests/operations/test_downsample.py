# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from loguru import logger
import numpy

import torch
import torch.nn as nn
import ttnn
import tt_lib as ttl
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0

# from tests.ttnn.integration_tests.resnet.resnet50 import write_to_file


def write_to_file(file_name, tensor):
    shape = tensor.shape
    height = shape[0] * shape[1] * shape[2]
    width = shape[3]

    tensor = tensor.reshape(height, width).float()
    tensor = tensor.cpu().detach().numpy()
    numpy.savetxt(file_name, tensor, fmt="%f", delimiter=",")


max_grid_size = (9, 12)


def get_sharded_config(input_2d_height):
    max_nshards = min(input_2d_height, max_grid_size[0] * max_grid_size[1])
    nshards = max_nshards
    while nshards > 0:
        if input_2d_height % nshards == 0:
            break
        nshards -= 1

    ncores = nshards
    if ncores % max_grid_size[1] == 0:
        grid_size = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
    else:
        if ncores < max_grid_size[1]:
            grid_size = ttnn.CoreGrid(y=1, x=ncores)
        else:
            grid1_size = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid2_size = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
            grid_size = [grid1_size, grid2_size]
    print(f" grid size --> {grid_size}")
    return grid_size


def get_block_sharded_config(batch_size, height, num_channels):
    max_nshards_h = min(batch_size * height, max_grid_size[0])  ## height along NHW
    max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
    ## find nshards_h along NHW
    nshards_h = max_nshards_h
    while nshards_h > 0:
        if batch_size * height % nshards_h == 0:
            break
        nshards_h -= 1
    ## find nshards_w along C
    nshards_w = max_nshards_w
    while nshards_w > 0:
        if num_channels % nshards_w == 0:
            break
        nshards_w -= 1

    if nshards_w == 0 or nshards_h == 0:
        raise ValueError("nshards_h or nshards_w is 0")

    ## calculate grid_size and shard_shape
    grid_size = ttnn.CoreGrid(y=nshards_h, x=nshards_w)
    return grid_size


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, stride_h, stride_w, num_cores, grid_size, height_sharded",
    [
        # [2, 256, 28, 28],
        (8, 64, 64, 56, 56, 2, 2, 98, (12, 9), True),
        # (8, 256, 256, 56, 56, 2, 2, 98, (12, 9), True),
        # (8, 512, 512, 28, 28, 2, 2, 80, (10, 8), False),
        # (8, 1024, 1024, 14, 14, 2, 2, 56, (7, 8), False),
    ],
)
# @pytest.mark.parametrize("scale_h", [2])
# @pytest.mark.parametrize("scale_w", [2])
# @pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT])
def test_downsample_multi_core(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    stride_h,
    stride_w,
    num_cores,
    grid_size,
    height_sharded,
):
    ## input shape is N C H W
    input_shape = (batch_size, input_channels, input_height, input_width)
    output_height = math.ceil(input_height / stride_h)
    output_width = math.ceil(input_width / stride_w)
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    ## calculated ttnn result
    ## permute to N H W C
    tt_input = input.permute(0, 2, 3, 1)
    dtype = ttnn.DataType.BFLOAT16
    tt_input = tt_input.reshape(1, 1, batch_size * input_height * input_width, input_channels)

    A_cl_host = ttnn.from_torch(tt_input, dtype, device=device)
    A_cl_host = ttnn.reshape(A_cl_host, (1, 1, batch_size * input_height * input_width, input_channels))
    # A_cl_host = A_cl_host.pad(input_shape, (0, 0, 0, 0), 0.0)

    mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    A_cl_host = ttnn.to_layout(A_cl_host, ttnn.TILE_LAYOUT)
    A_interleaved = A_cl_host  # ttnn.to_memory_config(A_cl_host, memory_config=mem_config)
    print(f"A_interleaved shape: {A_interleaved.shape}")
    input_2d_height = A_interleaved.shape[2]
    input_2d_width = A_interleaved.shape[3]

    print(f"input_2d_height={input_2d_height}, input_2d_width={input_2d_width}")

    num_cores_height_slices = num_cores if height_sharded else grid_size[0]
    input_2d_height_padded = _nearest_y(input_2d_height, num_cores_height_slices * 32)
    input_shard_height = (int)(input_2d_height_padded / num_cores_height_slices)
    input_shard_width = input_2d_width if height_sharded else ((int)(input_2d_width / grid_size[1]))

    sharded_memory_layout = ttnn.ShardStrategy.HEIGHT if height_sharded else ttnn.ShardStrategy.BLOCK
    sharded_memory_orientation = (
        ttnn.ShardOrientation.ROW_MAJOR if height_sharded else ttnn.ShardOrientation.COLUMN_MAJOR
    )

    # sharded_memory_orientation = (
    #     ttl.tensor.ShardOrientation.ROW_MAJOR if height_sharded else ttl.tensor.ShardOrientation.COL_MAJOR
    # )

    print(f"grid_size={grid_size}")
    print(f"shard_memory_layout={sharded_memory_layout}")
    print(f"input_shard_height={input_shard_height}, input_shard_width={input_shard_width}")

    tn_grid_size = ttnn.CoreGrid(y=grid_size[1], x=grid_size[0])
    # in_shard_shape = ttnn.ShardShape(y=input_shard_height, x=input_shard_width)
    # in_shard_shape = ttnn.ShardShape(x=input_shard_height, y=input_shard_width)

    # A_sharded = ttl.tensor.interleaved_to_sharded(
    #     A_interleaved,
    #     grid_size,
    #     [input_shard_height, input_shard_width],
    #     sharded_memory_layout,
    #     sharded_memory_orientation,
    # )

    # abhinav_grid_size = {}
    # if True: # shard_strategy == ttnn.ShardStrategy.HEIGHT:
    #     ## nsticks per shard should be divisible by in_w
    #     max_nshards = min(input_2d_height, max_grid_size[0] * max_grid_size[1])
    #     nshards = max_nshards
    #     while nshards > 0:
    #         if input_2d_height % nshards == 0:
    #             break
    #         nshards -= 1

    #     ncores = nshards
    #     if ncores % max_grid_size[1] == 0:
    #         abhinav_grid_size = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
    #     else:
    #         if ncores < max_grid_size[1]:
    #             abhinav_grid_size = ttnn.CoreGrid(y=1, x=ncores)
    #         else:
    #             grid1_size = (ncores // max_grid_size[1], max_grid_size[1])
    #             grid2_size = (ncores // max_grid_size[1] + 1, ncores % max_grid_size[1])
    #             abhinav_grid_size = ttnn.CoreGrid(y=grid1_size, x=grid2_size)

    #     #in_shard_shape = ttnn.ShardShape(y=input_2d_height // ncores, x=input_shard_width)  ## y, x
    #     print(f"ncores={ncores}, grid_size={abhinav_grid_size}, in_shard_shape={in_shard_shape}")
    #    #out_shard_shape = ttnn.ShardShape(y=batch_size * h * w * scale_h * scale_w // ncores, x=c)

    print("done")
    # abhinav_grid_size = get_sharded_config(input_2d_width, input_2d_height, ttnn.ShardStrategy.HEIGHT if height_sharded else ttnn.ShardStrategy.BLOCK)
    # print(f"abhinav_grid_size --> {abhinav_grid_size}&& tn grid size --> {tn_grid_size}")
    tn_grid_size = (
        get_sharded_config(input_shard_height) if height_sharded else get_block_sharded_config(1, input_shard_height, 1)
    )
    # in_sharded_memory = ttnn.create_sharded_memory_config(
    #     tn_grid_size, in_shard_shape, sharded_memory_layout, sharded_memory_orientation
    # )

    print(f"tn_grid_size --> {tn_grid_size}")
    in_sharded_memory = ttnn.create_sharded_memory_config(
        # [batch_size, input_2d_height, input_2d_width, input_channels],
        [input_shard_height, input_shard_width],
        tn_grid_size,
        sharded_memory_layout,
        use_height_and_width_as_shard_shape=height_sharded,
    )

    A_sharded = ttnn.to_memory_config(A_interleaved, memory_config=in_sharded_memory)
    downsample_params = [batch_size, input_height, input_width, stride_h, stride_w]
    A_downampled_sharded = ttnn.downsample(A_sharded, downsample_params, dtype=dtype)  # output_dtype=dtype)
    return
    output_tensor = ttnn.to_torch(A_downampled_sharded)

    write_to_file("actual.txt", tt_input)
    write_to_file("expected.txt", output_tensor)

    # for i in range(0, output_height *  output_width * batch_size):
    # for l in range(0, batch_size):
    for k in range(0, batch_size * output_height):
        for i in range(0, output_width):
            # for j in range(0, input_channels):
            # delta = output_tensor[0][0][output_width * k + i][j] -
            expected_output = tt_input[0][0][input_width * k * 2 + 2 * i]
            actual_output = output_tensor[0][0][output_width * k + i]
            assert_with_pcc(expected_output, actual_output)
            # if delta > 0.001:
            # print(f" error at {i} {j} {delta}")
            # exit()

    print("done")

    """
    scale_factor = (scale_h, scale_w, 1)
    input_tensor = ttnn.from_torch(tt_input, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.reshape(input_tensor, (1, 1, batch_size * h * w , c))
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    #input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    # parameter = [2, ]
    downsample_op_params = [batch_size, h, w, 2, 2]
    output_tensor = ttnn.downsample(input_tensor, downsample_op_params, memory_config=out_sharded_mem_config)
    # output_tensor = ttnn.upsample(input_tensor, scale_factor, memory_config=out_sharded_mem_config)
    # output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    # output_tensor = ttnn.to_torch(output_tensor)
    """

    # ## compare the results
    # torch_result = torch_result.permute(0, 2, 3, 1)
    # assert_with_pcc(torch_result, output_tensor)

    # allclose = torch.allclose(output_tensor, torch_result)
    # isclose = torch.all(torch.isclose(output_tensor, torch_result))
    # isequal = torch.equal(output_tensor, torch_result)
