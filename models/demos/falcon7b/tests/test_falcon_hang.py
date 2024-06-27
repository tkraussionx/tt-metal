from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch


def test_reproduce_lm_head_nd_32(
    device,
):
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    torch.manual_seed(1234)

    seq_len = 32
    a_shape = [1, 1, seq_len, 4544]
    b_shape = [1, 1, 4544, 65024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    b_t = ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)
    bias_t = None

    out = ttl.tensor.falcon_lm_head_matmul(a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype)

    ref_out = tt2torch_tensor(out)

    nd_output_count = 0

    for i in range(10000):
        out.deallocate(True)
        out = ttl.tensor.falcon_lm_head_matmul(
            a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype
        )

        pt_out = tt2torch_tensor(out)

        _, output_pcc = comp_pcc(ref_out, pt_out, 1)

        if output_pcc != 1:
            nd_output_count += 1

        logger.debug(f"Iteration = {i}, Output pcc={output_pcc}")

    print(f"Iterations with nd output: {nd_output_count}")
    assert nd_output_count == 0
