# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_eltwise_atanh_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    logger.info(
        f"Running atanh with input_shape {input_shape} dtype {dtype} dlayout {dlayout} input_mem_config {in_mem_config} output_mem_config {output_mem_config} data_seed {data_seed}"
    )

    try:
        # get ref result
        ref_value = torch.atanh(x)

        tt_result = ttnn_ops.atanh(
            x,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )

        assert_with_pcc(ref_value, tt_result, 0.99)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e


test_sweep_args = []

data_seeds = [
    19698958,
    11249810,
    6388054,
    8154922,
    7397428,
    12484268,
    5720419,
    11158061,
    4910975,
    7340822,
    1517803,
    19255748,
    4175638,
    6325702,
    19325774,
    4016313,
    3893862,
    1221114,
    726000,
    6529388,
]
i = 0
for shape in [(3, 10, 60, 12), (5, 60, 182), (4, 1, 52, 202), (38, 8), (7, 47, 96)]:
    for out_buffer_type in [ttnn.BufferType.DRAM, ttnn.BufferType.L1]:
        for in_buffer_type in [ttnn.BufferType.DRAM, ttnn.BufferType.L1]:
            test_sweep_args.append(
                (
                    [shape],
                    [ttnn.DataType.BFLOAT16],
                    [ttnn.Layout.ROW_MAJOR],
                    [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, in_buffer_type)],
                    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, out_buffer_type),
                    data_seeds[i],
                )
            )
            i += 1

print(len(test_sweep_args))


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_atanh(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_atanh_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
