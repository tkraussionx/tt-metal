# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "suite_dram_interleaved": {
        "batch_sizes": [(1,), (1,)],
        "height": [32, 32 * 32, 32 * 64, 32 * 512],
        "width": [32, 32 * 32, 32 * 64, 32 * 512],
        "dim": [-1, -2, -3],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "suite_l1_interleaved": {
        "batch_sizes": [(1,), (1,)],
        "height": [32, 32 * 32, 32 * 64, 32 * 512],
        "width": [32, 32 * 32, 32 * 64, 32 * 512],
        "dim": [-1, -2, -3],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
    },
    # "suite_l1_sharded": {
    #     "batch_sizes": [(1,), (2,)],
    #     "height": [32, 32*32, 32*64, 32*512],
    #     "width": [32, 32*32, 32*64, 32*512],
    #     "dim": [-1, -2, -3],
    #     "input_dtype": [ttnn.bfloat16],
    #     "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    #     "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    # },
    # "suite_2": {
    #     "batch_sizes": [(1,)],
    #     "height": [1024, 4096],
    #     "width": [1024, 2048],
    #     "broadcast": [None, "h", "hw"],
    #     "input_a_dtype": [ttnn.bfloat16],
    #     "input_b_dtype": [ttnn.bfloat16],
    #     "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
    #     "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
    #     "input_b_memory_config": [ttnn.L1_MEMORY_CONFIG],
    #     "input_a_memory_config": [ttnn.L1_MEMORY_CONFIG],
    #     "output_memory_config": [
    #         ttnn.MemoryConfig(
    #             ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    #             ttnn.BufferType.L1,
    #             ttnn.ShardSpec(
    #                 ttnn.CoreRangeSet(
    #                     {
    #                         ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(1, 4)),
    #                         ttnn.CoreRange(ttnn.CoreCoord(2, 3), ttnn.CoreCoord(2, 5)),
    #                     }
    #                 ),
    #                 [64, 64],
    #                 ttnn.ShardOrientation.ROW_MAJOR,
    #                 False,
    #             ),
    #         )
    #     ],
    # },
}


def run(
    batch_sizes, height, width, dim, input_dtype, input_memory_config, output_memory_config, *, device
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.softmax(torch_input_tensor, dim=dim)

    start_time = start_measuring_time()

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, memory_config=input_memory_config
    )

    output_tensor = ttnn.softmax(input_tensor, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
