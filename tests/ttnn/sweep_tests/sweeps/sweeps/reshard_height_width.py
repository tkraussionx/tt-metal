# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random
from tests.ttnn.sweep_tests.shard_utility import sharded_run


# parameters = {
#    "dtype": [ttnn.bfloat16],
#    "height": [16, 32],
#    "width": [16, 64, 512],
#    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
#    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#    "input_num_cores_x": [1, 2, 8],
#    "input_num_cores_y": [1, 4, 8],
#    "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
#    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
#    "output_num_cores_x": [1, 4, 8],
#    "output_num_cores_y": [1, 2, 8],
#    "output_shard_strategy": [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH],
# }

parameters = {
    "dtype": [ttnn.bfloat16],
    "height": [16],
    "width": [16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "input_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR],
    "input_num_cores_x": [2],
    "input_num_cores_y": [1],
    "input_shard_strategy": [ttnn.ShardStrategy.HEIGHT],
    "output_shard_orientation": [ttnn.ShardOrientation.ROW_MAJOR],
    "output_num_cores_x": [1],
    "output_num_cores_y": [1],
    "output_shard_strategy": [ttnn.ShardStrategy.HEIGHT],
}


def skip(
    layout,
    input_num_cores_x,
    input_num_cores_y,
    input_shard_strategy,
    output_num_cores_x,
    output_num_cores_y,
    output_shard_strategy,
    **_,
) -> Tuple[bool, Optional[str]]:
    input_num_cores = input_num_cores_x * input_num_cores_y
    output_num_cores = output_num_cores_x * output_num_cores_y

    if (
        layout == ttnn.ROW_MAJOR_LAYOUT
        and (input_num_cores == output_num_cores and input_num_cores != 1)
        and input_shard_strategy == ttnn.ShardStrategy.WIDTH
        and output_shard_strategy == ttnn.ShardStrategy.WIDTH
    ):
        return True, "Wormhole Watcher NOC write fail in reshard kernel"

    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def compute_lcm(x, y):
    # choose the greater number
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1

    return lcm


def roundTo32(x):
    if x < 32:
        return 32
    else:
        if x % 32 == 0:
            return x
        else:
            return ((x // 32) * 32) + 32


def run(
    dtype,
    height,
    width,
    layout,
    input_shard_orientation,
    input_num_cores_x,
    input_num_cores_y,
    input_shard_strategy,
    output_shard_orientation,
    output_num_cores_x,
    output_num_cores_y,
    output_shard_strategy,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    torch_input_tensor, output = sharded_run(
        dtype,
        height,
        width,
        layout,
        input_shard_orientation,
        input_num_cores_x,
        input_num_cores_y,
        input_shard_strategy,
        True,
        True,
        True,
        device,
        output_shard_orientation,
        output_num_cores_x,
        output_num_cores_y,
        output_shard_strategy,
    )
    return check_with_pcc(torch_input_tensor, output, 0.999)
