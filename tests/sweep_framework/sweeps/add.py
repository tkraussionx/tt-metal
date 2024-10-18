# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import tracy
import ttnn
import sys

from tests.ttnn.utils_for_testing import get_per_core_size_and_num_cores
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "suite_ajakovljevic_add_sharding_all": {
        "batch_sizes": [(1,)],
        # "height": [224, 256, 416, 512, 768, 1024, 1600, 2048],
        # "width": [224, 256, 416, 512, 768, 1024, 1600, 2048],
        "height": [256, 448, 1280, 1856],
        "width": [256, 448, 1280, 1856],
        "broadcast": [None],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT or test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Inputs to eltwise binary must be tilized"
    if test_vector["broadcast"] in {"w", "hw"} and test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Broadcasting along width is not supported for row major layout"
    return False, None


def return_block_sharded_shard_spec(height, width, num_cores):
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
            return output_shard_spec


def tracy_testing(
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
    device,
):
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)
    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    num_cores = 4

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)

    if input_a_memory_config != ttnn.DRAM_MEMORY_CONFIG and input_a_memory_config != ttnn.L1_MEMORY_CONFIG:
        input_a_memory_config.shard_spec = return_block_sharded_shard_spec(height, width, num_cores)

    if input_b_memory_config != ttnn.DRAM_MEMORY_CONFIG and input_b_memory_config != ttnn.L1_MEMORY_CONFIG:
        input_b_memory_config.shard_spec = return_block_sharded_shard_spec(height, width, num_cores)

    if output_memory_config != ttnn.DRAM_MEMORY_CONFIG and output_memory_config != ttnn.L1_MEMORY_CONFIG:
        output_memory_config.shard_spec = return_block_sharded_shard_spec(height, width, num_cores)

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


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
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
) -> list:
    profiler = tracy.Profiler()
    profiler.enable()

    tracy_testing(
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
        device,
    )
    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)
    profiler.disable()
    return [(True, "OK"), 0]
