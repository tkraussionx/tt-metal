# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import reduce_max_h as tt_reduce_max_h


def run_reduce_max_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    numbers = torch.arange(1, 45)
    result = numbers.unsqueeze(0).unsqueeze(0).expand(input_shape)
    x = result.bfloat16()
    # x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    print("tt x", x)
    # get ref result
    ref_value = pytorch_ops.reduce_max(x_ref, dims=(-2,))

    tt_result = tt_reduce_max_h(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    print("tt result", tt_result)
    # compare tt and golden outputs
    success, pcc_value = comp_equal(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 1, 44, 44),
        # (1, 1, 20, 116),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        16899236,
    ),
    #     (
    #         (1, 1, 20, 20),
    #         ttl.tensor.DataType.BFLOAT16,
    #         ttl.tensor.Layout.ROW_MAJOR,
    #         "SYSTEM_MEMORY",
    #         ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    #         2763978,
    #     ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_reduce_max_h_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_reduce_max_h_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
