# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_wormhole_b0, skip_for_grayskull

# Sequence:
# MM1
# ScaleMaskSoftmax
# MM2


@pytest.mark.parametrize(
    "seq_len",
    (32, 64, 128, 1024, 2048),
    ids=["seq_len_32", "seq_len_64", "seq_len_128", "seq_len_1024", "seq_len_2048"],
)
@pytest.mark.parametrize("num_cores", [64])
def test_falcon40B_attn_sequence(device, num_cores, seq_len):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    head_dim = 16
    query_layer_shape = [1, head_dim, seq_len, 64]
    attn_mask_tiny_shape = [1, 1, seq_len, seq_len]
    attn_mask_full_shape = [1, head_dim, seq_len, seq_len]
    key_transposed_shape = [1, 1, 64, seq_len]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 1, head_dim * seq_len, 64]

    scale = 0.125

    num_slices = 1
    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_attention_mask_tiny = torch.randn(attn_mask_tiny_shape).bfloat16().float()
    torch_attention_mask_full = torch_attention_mask_tiny.repeat(1, head_dim, 1, 1)
    torch_key_transposed = torch.randn(key_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.zeros(attention_output_shape).bfloat16().float()
    torch_attention_output2 = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    reference_attn_mask_full = torch2tt_tensor(
        torch_attention_mask_full,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    reference_attn_mask_tiny = torch2tt_tensor(
        torch_attention_mask_tiny,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    reference_key_transposed = torch2tt_tensor(
        torch_key_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    attention_output_reference_concatenated = torch2tt_tensor(
        torch_attention_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    attention_output_experimental_concatenated = torch2tt_tensor(
        torch_attention_output2,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT8_B,
    )

    tiles_per_shard = 1  # Works for 2k seq_len
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for i in range(num_slices):
        print("Running reference slice ", i, " ...")
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        attn_mask_slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_attn_mask_full,
            grid_size,
            mm_output_height_shard_spec,
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        #        if seq_len == 2048:
        #            subblock_w = 8  # best option
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm_slice = ttl.operations.primary.matmul(
            slice,
            reference_key_transposed,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=compute_kernel_config,
        )

        subblock_w = 1
        # if seq_len == 2048:
        #     subblock_w = 4
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttl.operations.primary.transformers.scale_mask_softmax_in_place(
            mm_slice, scale, attn_mask_slice, program_config=softmax_program_config, is_causal_mask=True
        )

        attn_mask_slice.deallocate(True)

        # subblock_w = 2
        # subblock_h = 1
        subblock_w = 1
        subblock_h = 1
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        attn_out_slice = ttl.operations.primary.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=compute_kernel_config,
        )

        ttl.tensor.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_reference_concatenated,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        slice.deallocate(True)
        mm_slice.deallocate(True)
        attn_out_slice.deallocate(True)

    for i in range(num_slices):
        print("Running experimental slice ", i, " ...")
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 4  # best option
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm_slice = ttl.operations.primary.matmul(
            slice,
            reference_key_transposed,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=compute_kernel_config,
        )

        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttl.operations.primary.transformers.scale_causal_mask_hw_dims_softmax_in_place(
            mm_slice, scale, reference_attn_mask_tiny, program_config=softmax_program_config
        )

        subblock_w = 2
        subblock_h = 1
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        attn_out_slice = ttl.operations.primary.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=ttl.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=compute_kernel_config,
        )

        ttl.tensor.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_experimental_concatenated,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        slice.deallocate(True)
        mm_slice.deallocate(True)
        attn_out_slice.deallocate(True)

    torch_out_reference = tt2torch_tensor(attention_output_reference_concatenated)
    torch_out_experimental = tt2torch_tensor(attention_output_experimental_concatenated)

    # Compare slice pcc
    passing = True
    slice_length = head_dim * seq_len // num_slices
    for slice_index in range(num_slices):
        print("Comparing slice ", slice_index, "...")
        slice_passing = False
        slice_passing, output = comp_pcc(
            torch_out_reference[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
            torch_out_experimental[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
        )
        passing = passing and slice_passing
        print("Slice PCC is: ", output)

    # Compare entire tensors as well
    entire_tensor_passing, output = comp_pcc(torch_out_reference, torch_out_experimental)
    passing = entire_tensor_passing and passing

    assert passing
