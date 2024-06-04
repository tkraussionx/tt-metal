# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import torch
import ttnn

from loguru import logger


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((32, 14336, 4096, 1, 2, 7, 1, 1, 10000), (32, 14336, 4096, 1, 2, 7, 1, 2, 10000)),
    ids=["ff2-pass", "ff2-hang"],
)
def test_mixtral_mlp_hang(
    t3k_device_mesh,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    reset_seeds,
):
    torch.manual_seed(1234)

    # ff2_in_prog_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    #     # compute_with_storage_grid_size=(6, 7),
    #     compute_with_storage_grid_size=(8, 8),
    #     in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
    #     out_subblock_h=1,  # Must be divisible by per_core_M
    #     out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    #     per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
    #     per_core_N=7,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
    #     fuse_batch=True,
    #     fused_activation=None,
    #     mcast_in0=True,
    # )

    # ff2_in_mem_config = ttnn.create_sharded_memory_config(
    #     shape=ff2_in_shape,
    #     core_grid=ttnn.CoreGrid(y=8, x=8),
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     # halo=False,
    #     use_height_and_width_as_shard_shape=True,
    # )

    ff2_out_prog_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,  # =7 K = 14336 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=out_subblock_h,  # =1 # Must be divisible by per_core_M
        out_subblock_w=out_subblock_w,  # =1 # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=per_core_M,  # =1  M / TILE_HEIGHT = 32 / 32
        per_core_N=per_core_N,  # =2  N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Shapes used in Mixtral8x7b
    ff2_in_shape = [1, 1, seq_len, inner_dim]  # [1, 1, 32, 14336]
    w2_shape = [1, 1, inner_dim, weights_n]  # [1, 1, 14336, 4096]

    A = torch.randn(ff2_in_shape)
    B = torch.randn(w2_shape)

    ff2_in_t = ttnn.from_torch(
        A,
        device=t3k_device_mesh,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_device_mesh),
    )
    # ff2_in_t = ttnn.to_device(ff2_in_t, t3k_device_mesh)

    w2 = ttnn.as_tensor(
        B,
        dtype=ttnn.bfloat8_b,
        device=t3k_device_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=0),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # w2 = ttnn.to_device(w2, t3k_device_mesh)

    # First run for a reference output
    ff2_out = ttnn.experimental.operations.primary.matmul_1d(
        ff2_in_t,
        w2,
        program_config=ff2_out_prog_config,
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        output_dtype=ttnn.bfloat8_b,
    )
    # loop_count iterations to test determinism/hang
    # On sjc-lab-t3002 with `ff2-hang` config hangs around iteration 6090 ~ 6099 .
    # With `ff2-pass' config if finishes the first 10k iterations. If run a second time independently of config, it hangs on iteration 4226
    for i in range(loop_count):
        logger.info(f"iteration: {i}")
        ff2_out.deallocate(True)
        ff2_out = ttnn.experimental.operations.primary.matmul_1d(
            ff2_in_t,
            w2,
            program_config=ff2_out_prog_config,
            output_mem_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            output_dtype=ttnn.bfloat8_b,
        )

    ff2_out.deallocate(True)
    w2.deallocate(True)
    ff2_in_t.deallocate(True)
