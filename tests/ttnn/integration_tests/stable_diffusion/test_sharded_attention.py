# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest

import tt_lib as ttl
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_attnention(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=128,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        k_slice = ttl.tensor.unpad(
            reference_key_layer_transposed,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), 63, seq_len - 1),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            slice,
            k_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        k_slice.deallocate()
        slice.deallocate()

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            im_data_format=ttl.tensor.DataType.BFLOAT16,
        )

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=128,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    attn_weights = ttl.tensor.bmm(
        reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
    )
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)
    attn_weights = ttl.tensor.bmm(attn_weights, reference_value_layer, output_mem_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing
