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


def run_backward_addcdiv_tests(
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
    z = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # other tensor1
    w = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)  # other tensor2

    try:
        # get ref result
        ref_value = torch.stack(pytorch_ops.addcdiv_bw(x, y, z, w, scalar))
        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        y = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[1], dtype[0])
        z = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[2], dtype[0])
        w = ttnn_ops.setup_ttnn_tensor(w, device, dlayout[0], in_mem_config[2], dtype[0])

        tt_result = ttnn.addcdiv_bw(x, y, z, w, scalar, memory_config=output_mem_config)

        tt_result = torch.stack(
            [
                ttnn_ops.ttnn_tensor_to_torch(tt_result[0]),
                ttnn_ops.ttnn_tensor_to_torch(tt_result[1]),
                ttnn_ops.ttnn_tensor_to_torch(tt_result[2]),
            ]
        )

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(5, 3, 55, 98)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (None),
        9765530,
        38.25,
    ),
    (
        [(1, 3, 40, 70)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (ttnn.L1_MEMORY_CONFIG),
        11324992,
        58.5,
    ),
    (
        [(3, 12, 25, 88)],
        [ttnn.bfloat16],
        [ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        (None),
        1196734,
        94.0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_backward_addcdiv(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    run_backward_addcdiv_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device)
