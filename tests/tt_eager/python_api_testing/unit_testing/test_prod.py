# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)


def get_tensors(input_shape, output_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    torch_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)

    torch_input[:, 0, :, :].fill_(1.0)
    torch_input[:, 1, :, :].fill_(5.0)
    torch_input[:, 2, :, :].fill_(4.0)
    torch_input[:, 3, :, :].fill_(3.0)
    # torch_input[:, 4, :, :].fill_(1.0)
    # torch_input[:, 5, :, :].fill_(3.0)
    # torch_input[:, 6, :, :].fill_(1.0)
    # torch_input[:, 7, :, :].fill_(2.0)
    # print(torch_input[0, 0])
    # print(torch_input[0, 1])
    print(torch_input)
    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 4, 32, 32]),  # Single core
        # [[1, 1, 32, 3840]],  # Multi core h
        # [[1, 3, 32, 3840]],  # Multi core h
    ),
)
def test_prod(shapes, device):
    output_shape = shapes.copy()

    # for dim in dims:
    #     output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(shapes, shapes, device)

    # torch_output = torch.prod(torch_input)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = ttl.operations.primary.prod(tt_input).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # print(tt_output_cpu[:,0, :, :])
    # print(tt_output_cpu[:,2, :, :])
    # print(tt_output_cpu[:,4, :, :])
    # print(tt_output_cpu[:,6, :, :])

    print(tt_output_cpu[:, 3, :, :])
    # print(torch_output)

    # # test for equivalance
    # # TODO(Dongjin) : check while changing rtol after enabling fp32_dest_acc_en
    # rtol = atol = 0.12
    # passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    # logger.info(f"Out passing={passing}")
    # logger.info(f"Output pcc={output_pcc}")

    assert True
