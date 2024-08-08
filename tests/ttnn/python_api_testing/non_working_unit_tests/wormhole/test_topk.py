from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def run_topk_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    k,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_values, ref_inds = torch.topk(x, k, dim=-1, largest=True, sorted=True)

        t0 = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_values, tt_indices = ttnn.topk(t0, k, dim=-1, largest=True, sorted=True)
        tt_values = ttnn_ops.ttnn_tensor_to_torch(tt_values)
        tt_indices = ttnn_ops.ttnn_tensor_to_torch(tt_indices).to(torch.int64)
        tt_indices = torch.gather(x, -1, tt_indices)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_values.shape) == len(ref_values.shape)
    assert tt_values.shape == ref_values.shape
    assert_with_pcc(ref_values, tt_values, 0.99)


test_sweep_args = [
    (
        [(2, 12, 64, 32)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        32,
        2482923,
    ),
    (
        [(2, 12, 64, 192)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        32,
        2482923,
    ),
    (
        [(4, 7, 32, 256)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.DRAM_MEMORY_CONFIG,
        64,
        17155532,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, k, data_seed",
    (test_sweep_args),
)
def test_topk(input_shape, dtype, dlayout, in_mem_config, out_mem_config, k, data_seed, device):
    run_topk_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, k, data_seed, device)
