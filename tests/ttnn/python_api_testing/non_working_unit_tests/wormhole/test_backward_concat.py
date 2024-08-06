# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_backward_concat_tests(
    grad_shape,
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=grad_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)
    z = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    dim = -1
    for i in range(0, len(grad_shape[0])):
        if x.size(i) == y.size(i) + z.size(i):
            dim = i

    try:
        # get ref result
        ref_value = pytorch_ops.concat_bw(x, y, z)

        t0 = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])
        t1 = ttnn_ops.setup_ttnn_tensor(y, device, dlayout[0], in_mem_config[0], dtype[0])
        t2 = ttnn_ops.setup_ttnn_tensor(z, device, dlayout[0], in_mem_config[0], dtype[0])

        t3 = ttnn.concat_bw(t0, t1, t2, dim=dim, memory_config=output_mem_config)

        tt_result = [
            ttnn_ops.ttnn_tensor_to_torch(t3[0], output_mem_config),
            ttnn_ops.ttnn_tensor_to_torch(t3[1], output_mem_config),
        ]

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result) == len(ref_value)
    assert len(tt_result[0].shape) == len(ref_value[0].shape)
    assert len(tt_result[1].shape) == len(ref_value[1].shape)
    assert tt_result[0].shape == ref_value[0].shape, f"Shape mismatch at the first return tensor, dim = {dim}"
    assert tt_result[1].shape == ref_value[1].shape, f"Shape mismatch at the second return tensor, dim = {dim}"
    assert_with_pcc(ref_value[0], tt_result[0], 0.99)
    assert_with_pcc(ref_value[1], tt_result[1], 0.99)


test_sweep_args = [
    (
        [(6, 8, 96, 160)],
        [(3, 8, 96, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
    (
        [(3, 16, 96, 160)],
        [(3, 8, 96, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
    (
        [(3, 8, 192, 160)],
        [(3, 8, 96, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
    (
        [(3, 8, 96, 320)],
        [(3, 8, 96, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        16934480,
    ),
]


@pytest.mark.parametrize(
    "grad_shape, input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_backward_concat(grad_shape, input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_backward_concat_tests(grad_shape, input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
