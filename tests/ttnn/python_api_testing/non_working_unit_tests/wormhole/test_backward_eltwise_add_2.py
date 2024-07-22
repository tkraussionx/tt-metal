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


def run_backward_eltwise_add_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # grad tensor
    y = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # input tensor
    z = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # other tensor

    try:
        # get ref result
        ref_value = torch.stack(pytorch_ops.add_bw(x, y, z))
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[1], dtype[0])
        z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[2], dtype[0])

        tt_result = ttnn.add_bw(x, y, z, memory_config=output_mem_config)

        tt_result = torch.stack(
            [ttnn_ops.ttnn_tensor_to_torch(tt_result[0]), ttnn_ops.ttnn_tensor_to_torch(tt_result[1])]
        )

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(4, 7, 32, 96)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, None, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        3181992,
    ),
    (
        [(4, 7, 32, 96)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        (ttnn.DRAM_MEMORY_CONFIG),
        8123768,
    ),
    (
        [(4, 7, 32, 96)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None],
        (ttnn.DRAM_MEMORY_CONFIG),
        13587334,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_eltwise_add(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_eltwise_add_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
