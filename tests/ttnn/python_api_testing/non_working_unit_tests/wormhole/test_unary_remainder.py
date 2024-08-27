# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from itertools import product

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_unary_remainder_tests(
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

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)

    if dtype[0] == ttnn.bfloat8_b:
        x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None)
        x = ttnn.to_torch(x)
    else:
        x = x.to(torch.bfloat16)

    try:
        # get ref result
        ref_value = pytorch_ops.unary_remainder(x, scalar=scalar)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.remainder(x, scalar, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(12, 224, 224)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        8184290,
        20.75,
    ),
    (
        [(2, 4, 224, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        15031364,
        47.0,
    ),
    (
        [(1, 10, 192, 96)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        4369607,
        0.010009765625,
    ),
    (
        [(10, 224, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        3893862,
        70.5,
    ),
    (
        [(64, 32)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        1321330,
        83.0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_unary_remainder(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    run_unary_remainder_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device)
