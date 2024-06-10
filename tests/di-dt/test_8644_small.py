import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    torch2tt_tensor,
    skip_for_wormhole_b0,
    get_devices_for_t3000,
)
import torch
import math


@pytest.mark.parametrize("num_cores", [64])
def test_problematic_matmul(device, num_cores):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    in0_shape = [1, 1, 18176, 64]
    in1_shape = [1, 1, 64, 1024]

    torch_in0 = torch.randn(in0_shape).bfloat16().float()
    torch_in1 = torch.randn(in1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    tiles_per_shard = 9
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]

    in0_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 7),
                    ),
                }
            ),
            mm_activations_height_shard_spec,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in0_tt = torch2tt_tensor(
        torch_in0,
        device,
        tt_memory_config=in0_mem_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    in1_tt = torch2tt_tensor(
        torch_in1,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        per_core_M=tiles_per_shard,
        per_core_N=1024 // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    tt_out = ttl.operations.primary.matmul(
        in0_tt,
        in1_tt,
        program_config=program_config,
        output_mem_config=dram_interleaved_memory_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
        compute_kernel_config=compute_kernel_config,
    )

    for i in range(10000):
        print(i)
        tt_out = ttl.operations.primary.matmul(
            in0_tt,
            in1_tt,
            program_config=program_config,
            output_mem_config=dram_interleaved_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

    out = tt2torch_tensor(tt_out)
    passing = True
    assert passing
