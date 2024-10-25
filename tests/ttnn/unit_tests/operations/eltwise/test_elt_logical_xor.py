# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn


from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_logical_xor_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand_inf(size=input_shape[0], low=-100, high=100)
    y = gen_rand_inf(size=input_shape[1], low=-100, high=100)

    # compute ref value
    ref_value = pytorch_ops.logical_xor(x, y)

    tt_result = ttnn_ops.logical_xor(
        x=x,
        y=y,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(4, 7, 32, 96), (4, 7, 32, 96)],
        [ttnn.bfloat16, ttnn.bfloat16],
        [ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT],
        [
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ],
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16305027,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_logical_xor2(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    random.seed(0)
    run_logical_xor_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
