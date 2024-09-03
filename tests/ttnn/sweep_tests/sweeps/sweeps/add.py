# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

# configs = generate_configurations(
#     [1, 2, 3, 4], [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], [ttnn.bfloat16, ttnn.bfloat8_b], 5
# )

BufferType = ttnn._ttnn.deprecated.tensor.BufferType
TensorMemoryLayout = ttnn._ttnn.deprecated.tensor.TensorMemoryLayout
MemoryConfig = ttnn._ttnn.deprecated.tensor.MemoryConfig

# cores_4x4_spec =  ttnn.CoreRangeSet(
#     {
#         ttnn.CoreRange(
#             ttnn.CoreCoord(0, 0),
#             ttnn.CoreCoord(4, 4),
#         ),
#     }
# )

shard_4x4_spec = (
    ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        ),
        [
            32,  # per_core_M * TILE_HEIGHT
            256,  # per_core_N * TILE_WIDTH
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
)

# L1_BLOCK_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)
# L1_HEIGHT_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
# L1_WIDTH__SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1)

parameters = {
    "batch_sizes": [(1,)],
    "height": [1024],
    "width": [1024],
    "broadcast": [None],  # [None, "h", "w", "hw"],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat16],
    "input_a_layout": [
        ttnn.TILE_LAYOUT
    ],  # ttnn.ROW_MAJOR_LAYOUT, not supported for DRAM_MEMORY_CONFIG and L1_MEMORY_CONFIG
    "input_b_layout": [ttnn.TILE_LAYOUT],
    # "input_a_memory_config": [ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG],
    # "input_b_memory_config": [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    # "output_memory_config":  [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG],
    "input_a_memory_config": [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ],
    "input_b_memory_config": [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ],
    "output_memory_config": [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ],
    # "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    # "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    # "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
}


def MemoryLayoutToShardStrategy(memory_layout):
    if memory_layout == TensorMemoryLayout.HEIGHT_SHARDED:
        return ttnn.ShardStrategy.HEIGHT
    elif memory_layout == TensorMemoryLayout.WIDTH_SHARDED:
        return ttnn.ShardStrategy.WIDTH
    elif memory_layout == TensorMemoryLayout.BLOCK_SHARDED:
        return ttnn.ShardStrategy.BLOCK
    else:
        return None


def skip(*, broadcast, input_b_layout, **_) -> Tuple[bool, Optional[str]]:
    if broadcast in {"w", "hw"} and input_b_layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Broadcasting along width is not supported for row major layout"
    return False, None


def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)
    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    shard_strategy_a = MemoryLayoutToShardStrategy(input_a_memory_config.memory_layout)
    if shard_strategy_a:
        input_a_memory_config = ttnn.create_sharded_memory_config(
            shape=input_shape_a,
            core_grid=ttnn.CoreGrid(x=4, y=4),
            strategy=shard_strategy_a,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # use_height_and_width_as_shard_shape=True,
        )
        print("A ::", input_shape_a, input_a_memory_config)

    shard_strategy_b = MemoryLayoutToShardStrategy(input_a_memory_config.memory_layout)
    if shard_strategy_b:
        input_b_memory_config = ttnn.create_sharded_memory_config(
            shape=input_shape_b,
            core_grid=ttnn.CoreGrid(x=4, y=4),
            strategy=shard_strategy_b,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # use_height_and_width_as_shard_shape=True,
        )
        print("B ::", input_shape_b, input_b_memory_config)

    output_shape = (*batch_sizes, height, width)
    shard_strategy_o = MemoryLayoutToShardStrategy(output_memory_config.memory_layout)
    if shard_strategy_o:
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=output_shape,
            core_grid=ttnn.CoreGrid(x=4, y=4),
            strategy=shard_strategy_o,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            # use_height_and_width_as_shard_shape=True,
        )
        print("O ::", output_shape, output_memory_config)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
