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


def run_eltwise_exp_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-88, 88).to(torch.bfloat16)
    # x = torch.ones(size=input_shape[0]).to(torch.bfloat16) * 82.6
    ttnn.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    tensor_str = str(x)
    with open("../torch_inp_100.txt", "w") as file:
        file.write(tensor_str)

    try:
        # get ref result
        ref_value = torch.exp(x)

        torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
        # print("ref value", ref_value)
        tensor_str = str(ref_value)
        with open("../torch_tensor_exp.txt", "w") as file:
            file.write(tensor_str)

        tensor_str = str(x)
        with open("../ttnn_inp_exp.txt", "w") as file:
            file.write(tensor_str)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        # print("ttnn input", x)

        tt_result = ttnn.exp(x)
        tt_result_rm = ttnn.to_layout(tt_result, ttnn.ROW_MAJOR_LAYOUT)

        # print("ttnn value", tt_result_rm)
        tensor_str = str(tt_result_rm)
        with open("../ttnn_tensor_exp.txt", "w") as file:
            file.write(tensor_str)

        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

        # print("tt value", tt_result)
        tensor_str = str(tt_result)
        with open("../tt_tensor_exp.txt", "w") as file:
            file.write(tensor_str)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(1, 1, 32, 32)],
        # [(3, 2, 192, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6861134,
    ),
    # (
    #     [(12, 224, 224)],
    #     [ttnn.bfloat8_b],
    #     [ttnn.TILE_LAYOUT],
    #     [ttnn.DRAM_MEMORY_CONFIG],
    #     ttnn.L1_MEMORY_CONFIG,
    #     6411147,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_exp(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_exp_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
