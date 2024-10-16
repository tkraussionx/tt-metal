# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import tracy

import ttnn

from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import get_per_core_size_and_num_cores


def testing(device):
    height = 1024
    width = 1024
    num_cores = 4
    input_shape_a = (1, 1024, 1024)
    input_shape_b = (1, 1024, 1024)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for per_core_height, num_cores_height in get_per_core_size_and_num_cores(
        height, (num_cores,), max_per_core_size=1024
    ):
        for per_core_width, num_cores_width in get_per_core_size_and_num_cores(
            width, (num_cores,), max_per_core_size=1024
        ):
            output_shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_width - 1, num_cores_height - 1))}
                ),
                (per_core_height, per_core_width),
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            )
            output_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            output_memory_config.shard_spec = output_shard_spec
            output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
            output_tensor = ttnn.to_torch(output_tensor)
            break
        break


if __name__ == "__main__":
    device = ttnn.CreateDevice(0)
    testing(device)
    ttnn.close_device(device)
