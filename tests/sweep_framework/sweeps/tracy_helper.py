# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import tracy

import ttnn

from models.utility_functions import torch_random


def testing(device):
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
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)


if __name__ == "__main__":
    profiler = tracy.Profiler()

    device = ttnn.CreateDevice(0)
    profiler.enable()
    testing(device)
    ttnn.DumpDeviceProfiler(device)
    ttnn.synchronize_device(device)
    profiler.disable()
    ttnn.close_device(device)
