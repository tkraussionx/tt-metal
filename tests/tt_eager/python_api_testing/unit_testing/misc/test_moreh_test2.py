# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from loguru import logger

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, data_npu_dtype, data_cpu_dtype, device):
    condition_npu_dtype = ttnn.DataType.INT8
    condition_cpu_dtype = torch.int8
    npu_layout = ttnn.Layout.TILE

    torch_condition = torch.randint(0, 2, input_shape, dtype=condition_cpu_dtype)
    torch_input = torch.rand(input_shape, dtype=data_cpu_dtype)
    torch_other = torch.rand(input_shape, dtype=data_cpu_dtype)
    torch_output = torch.zeros(input_shape, dtype=data_cpu_dtype)

    tt_condition = ttnn.Tensor(torch_condition, condition_npu_dtype).pad_to_tile(0).to(npu_layout).to(device)
    tt_input = ttnn.Tensor(torch_input, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_other = ttnn.Tensor(torch_other, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_condition, tt_input, tt_other, tt_output, torch_condition, torch_input, torch_other


def get_tensors_backward(input_shape, data_npu_dtype, data_cpu_dtype, device):
    condition_npu_dtype = ttnn.DataType.INT8
    condition_cpu_dtype = torch.int8
    npu_layout = ttnn.Layout.TILE

    torch_condition = torch.randint(0, 2, input_shape, dtype=condition_cpu_dtype)
    torch_output_grad = torch.rand(input_shape, dtype=data_cpu_dtype)
    torch_input_grad = torch.zeros(input_shape, dtype=data_cpu_dtype)
    torch_other_grad = torch.zeros(input_shape, dtype=data_cpu_dtype)

    tt_condition = ttnn.Tensor(torch_condition, condition_npu_dtype).pad_to_tile(0).to(npu_layout).to(device)
    tt_output_grad = ttnn.Tensor(torch_output_grad, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_input_grad = ttnn.Tensor(torch_input_grad, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_other_grad = ttnn.Tensor(torch_other_grad, data_npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_condition, tt_output_grad, tt_input_grad, tt_other_grad, torch_condition, torch_output_grad


@pytest.mark.parametrize(
    "input_shape",
    (
        ([TILE_HEIGHT, TILE_WIDTH]),
        # ([TILE_HEIGHT // 2, TILE_WIDTH // 2]),
        # ([2, 3, 4, TILE_HEIGHT * 5 + TILE_HEIGHT // 2, TILE_WIDTH * 5 + TILE_WIDTH // 2]),
    ),
    ids=[
        "0",
        # "1",
        # "2",
    ],
)
@pytest.mark.parametrize(
    "npu_dtype, cpu_dtype",
    [
        (ttnn.DataType.BFLOAT16, torch.bfloat16),
        (ttnn.DataType.FLOAT32, torch.float32),
    ],
    ids=[
        "bfloat16",
        "float32",
    ],
)
def test_moreh_where(input_shape, npu_dtype, cpu_dtype, device):
    torch.manual_seed(2023)

    (tt_condition, tt_input, tt_other, tt_output, torch_condition, torch_input, torch_other) = get_tensors(
        input_shape, npu_dtype, cpu_dtype, device
    )

    torch_output = torch.where(torch_condition > 0, torch_input, torch_other)
    cpu_layout = ttnn.Layout.ROW_MAJOR
    ttnn.experimental.operations.primary.moreh_test2(tt_input, tt_condition, tt_other, output=tt_output)
    tt_output_cpu = tt_output.cpu().to(cpu_layout).unpad_from_tile(input_shape).to_torch()

    # torch.set_printoptions(threshold=1000000, linewidth=100000000, sci_mode=False, precision=2)
    # print (tt_other)
    # print (torch_output)
    # print (tt_output_cpu)
    # print (torch_output - tt_output_cpu)
    assert torch.equal(torch_output, tt_output_cpu)
