# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_backward_add_unary_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    scalar,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # grad tensor
    y = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # input tensor

    try:
        # get ref result
        ref_value = pytorch_ops.unary_add_bw(x, y, scalar)
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[1], dtype[0])

        tt_result = ttnn.add_bw(x, y, alpha=scalar, memory_config=output_mem_config)[0]

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(4, 7, 32, 96)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, None],
        (ttnn.DRAM_MEMORY_CONFIG),
        11871267,
        36.0,
    ),
    (
        [(2, 5, 49, 50)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, None],
        (ttnn.DRAM_MEMORY_CONFIG),
        8726038,
        90.5,
    ),
    (
        [(4, 10, 72, 116)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [None, ttnn.L1_MEMORY_CONFIG],
        (ttnn.L1_MEMORY_CONFIG),
        11158061,
        53.0,
    ),
    (
        [(1, 2, 202, 38)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [None, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        4175638,
        81.0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_backward_add_unary(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    run_backward_add_unary_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device)
