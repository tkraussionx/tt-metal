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


def run_complex_polar_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.complex(
        torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16).to(torch.float),
        torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16).to(torch.float),
    )

    try:
        # get ref result
        ref_value = torch.polar(x.real, x.imag)

        x = ttnn.complex_tensor(
            ttnn_ops.setup_ttnn_tensor(x.real, device, dlayout[0], in_mem_config[0], dtype[0]),
            ttnn_ops.setup_ttnn_tensor(x.imag, device, dlayout[0], in_mem_config[0], dtype[0]),
        )
        tt_result = ttnn.polar(x, memory_config=output_mem_config)

        tt_result = torch.complex(
            ttnn_ops.ttnn_tensor_to_torch(tt_result.real).to(torch.float32),
            ttnn_ops.ttnn_tensor_to_torch(tt_result.imag).to(torch.float32),
        )

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.real.shape) == len(ref_value.real.shape)
    assert tt_result.real.shape == ref_value.real.shape
    assert_with_pcc(ref_value.real, tt_result.real, 0.99)

    assert len(tt_result.imag.shape) == len(ref_value.imag.shape)
    assert tt_result.imag.shape == ref_value.imag.shape
    assert_with_pcc(ref_value.imag, tt_result.imag, 0.99)


test_sweep_args = [
    (
        [(224, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        14112040,
    ),
    (
        [(6, 160, 64)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        9456908,
    ),
    (
        [(3, 2, 192, 64)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        11871267,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_complex_polar(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_complex_polar_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
