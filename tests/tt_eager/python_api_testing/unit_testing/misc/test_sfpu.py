# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


import tt_lib as ttl
import numpy as np
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0
import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from loguru import logger


def get_tt_tensor(torch_tensor, device, *, npu_dtype=ttl.tensor.DataType.BFLOAT16):
    npu_layout = ttl.tensor.Layout.TILE
    tt_tensor = ttl.tensor.Tensor(torch_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return tt_tensor


def to_cpu(npu_tensor, shape, *, cpu_layout=ttl.tensor.Layout.ROW_MAJOR):
    if npu_tensor is None:
        return None
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


def test_sfpu_with_uint32(device):
    shape = [1, 1, 32, 32]
    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    torch_input = torch.randint(3000000000, 3100000000, shape)
    tt_input = get_tt_tensor(torch_input, device, npu_dtype=ttl.tensor.DataType.UINT32)

    tt_output = ttl.tensor.identity_uint32(tt_input, output_mem_config=interleaved_mem_config)
    logger.debug(f"tt_output={tt_output}")


@pytest.mark.parametrize(
    "data_format",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        #  ttl.tensor.DataType.FLOAT32,
    ),
    ids=[
        "BFLOAT16",
        "BFLOAT8_B",
        #  "FLOAT32"
    ],
)
def test_sfpu_identity(data_format, device):
    # use single-tile
    shape = [1, 1, 32, 32]
    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    # prepare tensor
    torch_input = torch.randn(shape, dtype=torch.float)
    tt_input = get_tt_tensor(torch_input, device, npu_dtype=data_format)

    # identity api
    tt_output = ttl.tensor.identity(tt_input, output_mem_config=interleaved_mem_config)

    # tt_input == tt_output
    tt_input_cpu = to_cpu(tt_input, shape)
    tt_output_cpu = to_cpu(tt_output, shape)
    passing, output_pcc = comp_allclose_and_pcc(tt_input_cpu, tt_output_cpu)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing


@pytest.mark.parametrize(
    "data_format",
    (
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.UINT32,
    ),
    ids=[
        "UINT16",
        "UINT32",
    ],
)
def test_sfpu_identity_uint(data_format, device):
    # use single-tile
    shape = [1, 1, 32, 32]
    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    # prepare tensor
    torch_input = torch.randint(0, 10000, shape, dtype=torch.long)
    tt_input = get_tt_tensor(torch_input, device, npu_dtype=data_format)

    # identity api
    if data_format == ttl.tensor.DataType.UINT16:
        tt_output = ttl.tensor.identity(tt_input, output_mem_config=interleaved_mem_config)
    else:
        tt_output = ttl.tensor.identity_uint32(tt_input, output_mem_config=interleaved_mem_config)

    tt_input_cpu = to_cpu(tt_input, shape)
    tt_output_cpu = to_cpu(tt_output, shape)
    passing, output_pcc = comp_allclose_and_pcc(tt_input_cpu, tt_output_cpu)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")
    breakpoint()
    assert passing


def print_hex_or_bin(tensor, is_numpy, is_hex):
    for i in range(0, 8):
        if is_numpy == True:
            val = hex(tensor[0][0][0][i]) if is_hex else bin(tensor[0][0][0][i])
        else:
            val = hex(tensor[0][0][0][i].item()) if is_hex else bin(tensor[0][0][0][i].item())
        print(f"{val:>5} ", end="")
    print("")


shape = [1, 1, 32, 32]
interleaved_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)


def moreh_sfpu_test(test_case, device):
    torch.manual_seed(3062)
    torch_input = torch.arange(1024)
    torch_input = torch_input.reshape(shape)
    torch_output = torch.ones(shape)
    tt_input = get_tt_tensor(torch_input, device, npu_dtype=ttl.tensor.DataType.UINT32)
    tt_output = get_tt_tensor(torch_output, device, npu_dtype=ttl.tensor.DataType.BFLOAT16)

    logger.debug(f"before running SFPU")
    logger.debug(f"tt_input {tt_input}")
    logger.debug(f"tt_output {tt_output}")

    ttl.operations.primary.moreh_sfpu_test(tt_input, tt_output, test_case)
    tt_input_cpu = to_cpu(tt_input, shape)
    tt_output_cpu = to_cpu(tt_output, shape)

    logger.debug(f"after running SFPU")
    logger.debug(f"tt_input {tt_input}")
    logger.debug(f"tt_output {tt_output}")


def test_sfpu_bfloat16_output_is_weird(device):
    moreh_sfpu_test(0, device)


def test_sfpu_bfloat16_output_is_ok(device):
    moreh_sfpu_test(1, device)


def test_copy_tiles(device):
    moreh_sfpu_test(2, device)
