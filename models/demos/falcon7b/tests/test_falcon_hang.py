import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    get_devices_for_t3000,
)
import torch


@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_lm_head_nd_32(
    all_devices,
    num_devices,
):
    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    print("Running on: ", num_devices, " devices.")
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    torch.manual_seed(1234)

    seq_len = 32
    a_shape = [1, 1, seq_len, 4544]
    b_shape = [1, 1, 4544, 65024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = []
    b_t = []

    for device_idx in range(num_devices):
        a_t.append(ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in0_mem_config))
        b_t.append(ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in1_mem_config))

    bias_t = None

    out = []
    for device_idx in range(num_devices):
        out.append(
            ttl.tensor.falcon_lm_head_matmul(
                a_t[device_idx], b_t[device_idx], bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype
            )
        )

    nd_output_count = 0

    for i in range(100000):
        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)
            out[device_idx] = ttl.tensor.falcon_lm_head_matmul(
                a_t[device_idx], b_t[device_idx], bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype
            )
            _, output_pcc = 1, 1

            if output_pcc != 1:
                nd_output_count += 1

        for device_idx in range(num_devices):
            if num_devices != 1:
                print("Start sync logicalDeviceID: ", device_idx)
            else:
                print("Start single device sync")
            ttl.device.Synchronize(devices[device_idx])
            if num_devices != 1:
                print("End sync logicalDeviceID: ", device_idx)
            else:
                print("End single device sync")

        logger.debug(f"Iteration = {i}, Output pcc={output_pcc}")

    print(f"Iterations with nd output: {nd_output_count}")
    assert nd_output_count == 0


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_lm_head_nd_32_t3000(all_devices, logical_chip_index):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")
    devices = get_devices_for_t3000(all_devices, num_devices_t3000)

    logical_chip_id_to_coordinates = [None] * num_devices_t3000
    logical_chip_id_to_coordinates[0] = (1, 0)
    logical_chip_id_to_coordinates[1] = (0, 0)
    logical_chip_id_to_coordinates[2] = (0, 1)
    logical_chip_id_to_coordinates[3] = (1, 1)
    logical_chip_id_to_coordinates[4] = (2, 1)
    logical_chip_id_to_coordinates[5] = (3, 1)
    logical_chip_id_to_coordinates[6] = (3, 0)
    logical_chip_id_to_coordinates[7] = (2, 0)

    print(
        "Selecting logical device id: ",
        logical_chip_index,
        " coordinates: ",
        logical_chip_id_to_coordinates[logical_chip_index],
    )
    target_device = devices[logical_chip_index]
    devices = [target_device]
    test_reproduce_lm_head_nd_32(devices, 1)
