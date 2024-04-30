from loguru import logger
import pytest

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor
import torch


def fill_tile_constants(tensor_shape):
    N, C, H, W = tensor_shape
    Z = (N * C * H * W) // 1024
    tile_constant = 0
    tensor_constants = torch.ones(tensor_shape).reshape(1, Z, 32, 32)
    for z in range(0, Z):
        tensor_constants[0][z] = tile_constant
        tile_constant += 1

    return tensor_constants.reshape(tensor_shape)


def check_elems(device_tensor, torch_tensor):
    N, C, H, W = device_tensor.shape

    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if device_tensor[n][c][h][w] != torch_tensor[n][c][h][w]:
                        print(f"device_tensor[n][c][h] = {device_tensor[n][c][h]}")
                        print(f"torch_tensor[n][c][h] = {torch_tensor[n][c][h]}")
                        throw("Error")


@pytest.mark.parametrize(
    "seq_len, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (
        # (32, 8128, 1, 4, 4, 1, 4, 10000),  # lm head
        (128, 8128, 4, 4, 4, 1, 4, 20000),  # lm head
        # (32, 4096, 1, 2, 4, 1, 2, 10000), # mlp 4h
        # (128, 4096, 4, 2, 4, 2, 2, 20000), # mlp 4h
    ),
    ids=["lm_seq_len32"],
)
def test_reproduce_matmul_1d(
    device, seq_len, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count
):
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    torch.manual_seed(0)

    a_shape = [1, 1, seq_len, 8192]
    b_shape = [1, 1, 8192, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    a_t = ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    b_t = ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)
    bias_t = None

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttl.operations.primary.matmul_1d(
        a_t,
        b_t,
        program_config=program_config,
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
        compute_kernel_config=compute_config,
    )

    ref_out = tt2torch_tensor(out)

    nd_output_count = 0
    device_output = None

    for i in range(loop_count):
        out.deallocate(True)

        out = ttl.operations.primary.matmul_1d(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_config,
        )

        pt_out = tt2torch_tensor(out)
        if i == 0:
            device_output = pt_out

        _, output_pcc = comp_pcc(device_output, ref_out, 1)
        # check_elems(device_output, ref_out)

        if output_pcc != 1:
            nd_output_count += 1

        logger.debug(f"Iteration = {i}, Output pcc={output_pcc}")

    print(f"Iterations with nd output: {nd_output_count}")

    assert nd_output_count == 0
