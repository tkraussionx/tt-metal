import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0
import torch
import math
import time


@pytest.mark.parametrize("num_cores", [64])
def test_deallocate_full(device, num_cores):
    query_layer_shape = [1, 1, (71 * 1024) // 4, 32]
    key_layer_shape = [1, 1, 32, 1024]
    value_layer_shape = [1, 1, 1024, 32]
    mm_out_shape = [1, 1, 71 * 1024 // 4, 32]

    torch.manual_seed(0)
    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer = torch.randn(key_layer_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    # TT tensors
    print("Creating query layer")
    tt_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    print("Done creating query layer")
    print("Creating key layer")
    tt_key_layer = torch2tt_tensor(
        torch_key_layer, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    print("Done creating key layer")
    print("Creating value layer")
    tt_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    print("Done creating value layer")
    mm2_out_torch = torch.randn(mm_out_shape).bfloat16().float()
    print("Sending mm2out to device")
    mm2_out = torch2tt_tensor(
        mm2_out_torch, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    print("Done sending mm2out to device")
    print("Creating ref_mm1")
    ref_mm1 = ttl.tensor.matmul(tt_query_layer, tt_key_layer)
    print("Done Creating ref_mm1")
    ref_mm2 = ttl.tensor.matmul(ref_mm1, tt_value_layer)
    print("Done creating ref_mm2")
    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    height_shard_spec_1 = [9 * 32, 1 * 32]
    print("Creating query_layer_sharded")
    query_layer_sharded = ttl.tensor.interleaved_to_sharded(
        tt_query_layer,
        (8, 8),
        height_shard_spec_1,
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )
    print("Done creating query_layer_sharded")
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    program_config_1 = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        per_core_M=9,
        per_core_N=32,
        out_subblock_h=1,
        out_subblock_w=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    print("Making mm1_sharded")
    mm1_sharded = ttl.operations.primary.matmul(
        query_layer_sharded,
        tt_key_layer,
        program_config=program_config_1,
        output_mem_config=height_sharded_memory_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
        compute_kernel_config=compute_kernel_config,
    )
    print("Done making mm1_sharded")
    # If deallocate is moved after the next matmul, the test passes
    print("Deallocating query_layer")
    query_layer_sharded.deallocate()
    print("Done deallocating query_layer")
    program_config_2 = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=32,
        per_core_M=9,
        per_core_N=1,
        out_subblock_h=1,
        out_subblock_w=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    print("Making mm2_sharded")
    mm2_sharded = ttl.operations.primary.matmul(
        mm1_sharded,
        tt_value_layer,
        program_config=program_config_2,
        output_mem_config=height_sharded_memory_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
        compute_kernel_config=compute_kernel_config,
    )
    print("Done making mm2_sharded")
    # If deallocate is here, the test passes
    # query_layer_sharded.deallocate()

    mm2_out = ttl.tensor.sharded_to_interleaved(mm2_sharded, dram_interleaved_memory_config)

    # Compare results
    torch_ref_mm2 = tt2torch_tensor(ref_mm2)
    torch_mm2_out = tt2torch_tensor(mm2_out)

    passing, output = comp_pcc(
        torch_ref_mm2,
        torch_mm2_out,
    )

    print(output)
    assert passing


@pytest.mark.parametrize("num_cores", [64])
def test_deallocate_slice(device, num_cores):
    query_layer_shape = [1, 1, (71 * 1024) // 2, 32]
    key_layer_shape = [1, 1, 32, 1024]
    value_layer_shape = [1, 1, 1024, 32]
    mm_out_shape = [1, 1, 71 * 1024 // 2, 32]

    torch.manual_seed(0)
    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer = torch.randn(key_layer_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    # TT tensors
    tt_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    tt_key_layer = torch2tt_tensor(
        torch_key_layer, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )

    tt_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    mm2_out_torch = torch.randn(mm_out_shape).bfloat16().float()
    mm2_out = torch2tt_tensor(
        mm2_out_torch, device, tt_memory_config=dram_interleaved_memory_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )

    ref_mm1 = ttl.tensor.matmul(tt_query_layer, tt_key_layer)
    ref_mm2 = ttl.tensor.matmul(ref_mm1, tt_value_layer)

    # do the same with sharding
    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    height_shard_spec_1 = [9 * 32, 1 * 32]
    num_slices = 2
    for slice_index in range(num_slices):
        query_layer_slice_sharded = ttl.tensor.interleaved_to_sharded_partial(
            tt_query_layer,
            (8, 8),
            height_shard_spec_1,
            num_slices,
            slice_index,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        program_config_1 = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            per_core_M=9,
            per_core_N=32,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm1_sharded = ttl.operations.primary.matmul(
            query_layer_slice_sharded,
            tt_key_layer,
            program_config=program_config_1,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        # If deallocate is moved after the next matmul, the test passes
        query_layer_slice_sharded.deallocate()

        program_config_2 = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=32,
            per_core_M=9,
            per_core_N=1,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm2_sharded = ttl.operations.primary.matmul(
            mm1_sharded,
            tt_value_layer,
            program_config=program_config_2,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        # If deallocate is here, it works
        # query_layer_slice_sharded.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm2_sharded, mm2_out, num_slices, slice_index, dram_interleaved_memory_config
        )

    # Compare results
    torch_ref_mm2 = tt2torch_tensor(ref_mm2)
    torch_mm2_out = tt2torch_tensor(mm2_out)

    slice_to_compare = 0
    passing = True
    for slice_to_compare in range(num_slices):
        slice_passing = True
        slice_passing, output = comp_pcc(
            torch_ref_mm2[:, :, 18176 * slice_to_compare : 18176 * (slice_to_compare + 1), :],
            torch_mm2_out[:, :, 18176 * slice_to_compare : 18176 * (slice_to_compare + 1), :],
        )
        passing = passing and slice_passing
        print("Slice ", slice_to_compare, " PCC: ", output)

    assert passing
