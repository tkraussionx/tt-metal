# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger

import threading

# def my_task(item):
#     # Do some processing with the item
#     return item * 2

# # List of items to process
# items = [1, 2, 3, 4, 5]

# # Create a ThreadPoolExecutor with 4 threads
# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     # Submit the tasks to the executor
#     # Each task will process one item from the list
#     # and return the result
#     results = list(executor.map(my_task, items))

# print(results)


def launch_device_operation(torch_input_tensors, device_output_tensors, device_index, device):
    device_input_tensor_a = ttnn.from_torch(torch_input_tensors[device_index], layout=ttnn.TILE_LAYOUT, device=device)
    device_input_tensor_b = ttnn.from_torch(torch_input_tensors[device_index], layout=ttnn.TILE_LAYOUT, device=device)
    device_output_tensors[device_index] = device_input_tensor_a + device_input_tensor_b


@pytest.mark.parametrize("size", [64])
def test_add_1D_tensor_and_scalar(pcie_devices, size):
    torch.manual_seed(0)

    num_devices = len(pcie_devices)

    torch_output_tensors = [None] * num_devices
    torch_input_tensors = [None] * num_devices
    device_output_tensors = [None] * num_devices
    threads = [None] * num_devices

    for idx, device in enumerate(pcie_devices):
        torch_input_tensors[idx] = torch.ones((size,), dtype=torch.bfloat16) * (idx + 1)
        torch_output_tensors[idx] = torch_input_tensors[idx] + torch_input_tensors[idx]

    # Create threads
    for idx, device in enumerate(pcie_devices):
        threads[idx] = threading.Thread(
            target=launch_device_operation, args=(torch_input_tensors, device_output_tensors, idx, device)
        )

    # Start threads
    for thread in threads:
        thread.start()

    # Join threads
    for thread in threads:
        thread.join()

    for idx, device in enumerate(pcie_devices):
        device_output_tensor = ttnn.to_torch(device_output_tensors[idx], torch_rank=1)
        cpu_output_tensor = torch_output_tensors[idx]
        logger.info(f"Device id: {idx} Expected value is: {cpu_output_tensor[0]}")
        assert_with_pcc(cpu_output_tensor, device_output_tensor, 0.9999)
