# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_eltwise_relu_min_tests(
    input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, lower_limit, device
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)

    try:
        # get ref result
        ref_value = pytorch_ops.relu_min(x, lower_limit=lower_limit)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.relu_min(x, lower_limit, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)
        logger.info(f"Finished running relu_min")

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(6, 7, 216, 182)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        17601780,
        71.5,
    ),
    (
        [(2, 11, 152, 212)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        19451336,
        64.0,
    ),
    (
        [(4, 11, 166, 180)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        10882535,
        95.5,
    ),
    (
        [(4, 10, 163, 86)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        9094524,
        16.375,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, lower_limit",
    (test_sweep_args),
)
def test_eltwise_relu_min(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, lower_limit, device):
    run_eltwise_relu_min_tests(
        input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, lower_limit, device
    )
