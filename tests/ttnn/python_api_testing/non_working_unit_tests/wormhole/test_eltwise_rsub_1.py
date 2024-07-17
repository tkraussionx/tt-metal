from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops


def run_eltwise_rsub_tests(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, factor, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100)

    try:
        # get ref result
        ref_value = pytorch_ops.eltwise_rsub(x, factor=factor)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.rsub(x, factor, memory_config=output_mem_config)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result)
        logger.info(f"Finished running rsub")

    except Exception as e:
        logger.warning(f"Operation execution crashed")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    assert_with_pcc(ref_value, tt_result, 0.99)


test_sweep_args = [
    (
        [(7, 14, 32, 160)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [None],
        ttnn.DRAM_MEMORY_CONFIG,
        4673250,
        9.190519884672804,
    ),
    (
        [(10, 9, 480, 288)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [None],
        ttnn.DRAM_MEMORY_CONFIG,
        14568389,
        8.998499050883135,
    ),
    (
        [(12, 13, 384, 448)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [None],
        ttnn.L1_MEMORY_CONFIG,
        18361995,
        9.178487794238004,
    ),
    (
        [(4, 22, 160, 288)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        None,
        12321236,
        8.726966241937092,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor",
    (test_sweep_args),
)
def test_eltwise_rsub(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device):
    run_eltwise_rsub_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, factor, device)
