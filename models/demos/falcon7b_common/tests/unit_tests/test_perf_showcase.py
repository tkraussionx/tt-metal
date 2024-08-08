# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from enum import Enum, auto
from loguru import logger

import ttnn
from models.demos.falcon7b_common.tt.falcon_causallm import falcon_lm_head_matmul
from models.demos.falcon7b_common.tt.falcon_mlp import falcon_dense_4h_to_h_matmul, falcon_dense_h_to_4h_matmul
from models.demos.falcon7b_common.tt.model_utils import get_falcon_default_core_grid
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0
import torch
import math


@pytest.mark.parametrize("seq_len", [1024], ids=["seq_len_1024"])
@pytest.mark.parametrize("num_cores", [64])
def test_v0_sharded_sequence(
    device,
    seq_len,
    num_cores,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    num_heads = 64

    if seq_len == 128:
        num_slices = 1
    elif seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()
    torch_scalar = (torch.ones(scalar_shape) * (1 / math.sqrt(num_heads))).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )

    compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    passing = True
    output = None

    attention_output_concatenated = torch2tt_tensor(
        torch_attention_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for i in range(num_slices):
        slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8  # best option
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

        mm_slice = ttnn.matmul(
            slice,
            reference_key_layer_transposed,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        mm_slice = ttnn.experimental.operations.primary.bcast(
            mm_slice,
            reference_scalar,
            ttnn.experimental.tensor.BcastOpMath.MUL,
            ttnn.experimental.tensor.BcastOpDim.HW,
            output_mem_config=height_sharded_memory_config,
            in_place=True,
        )

        # Deallocating here causes pcc to drop - issue #6638
        # So we have to move it after the entire sequence is finished
        # slice.deallocate()

        attn_mask_slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
            attention_mask,
            grid_size,
            mm_output_height_shard_spec,
            num_slices,
            i,
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        )

        mm_slice = ttnn.add_(
            mm_slice,
            attn_mask_slice,
            activations=None,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
        )

        attn_mask_slice.deallocate()

        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttnn.softmax_in_place(
            mm_slice, program_config=softmax_program_config, compute_kernel_config=compute_kernel_config
        )

        subblock_w = 2
        subblock_h = 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

        attn_out_slice = ttnn.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        ttnn.experimental.tensor.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_concatenated,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        slice.deallocate()
        mm_slice.deallocate()
        attn_out_slice.deallocate()

    attention_output_concatenated_torch = tt2torch_tensor(attention_output_concatenated)

    attn_weights = ttnn.matmul(
        reference_query_layer, reference_key_layer_transposed, memory_config=dram_interleaved_memory_config
    )

    attn_weights = ttnn.experimental.operations.primary.bcast(
        attn_weights,
        reference_scalar,
        ttnn.experimental.tensor.BcastOpMath.MUL,
        ttnn.experimental.tensor.BcastOpDim.HW,
        output_mem_config=dram_interleaved_memory_config,
    )
    attn_weights = ttnn.add(attn_weights, attention_mask, memory_config=dram_interleaved_memory_config)
    attn_weights = ttnn.softmax_in_place(attn_weights, compute_kernel_config=compute_kernel_config)
    attn_output = ttnn.matmul(attn_weights, reference_value_layer)
    attn_output_torch = tt2torch_tensor(attn_output)
    passing = True

    attn_output_torch_reshaped = attn_output_torch.view(1, 1, 71 * seq_len, 64)
    attention_output_concatenated_torch_reshaped = attention_output_concatenated_torch.view(1, 1, 71 * seq_len, 64)
    slice_length = (71 * seq_len) // num_slices
    for slice_index in range(num_slices):
        print("Comparing slice ", slice_index, "...")
        slice_passing = False
        slice_passing, output = comp_pcc(
            attn_output_torch_reshaped[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
            attention_output_concatenated_torch_reshaped[
                :, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :
            ],
        )
        passing = passing and slice_passing
        print("Slice PCC is: ", output)

    # Compare entire tensors as well
    entire_tensor_passing, output = comp_pcc(attn_output_torch, attention_output_concatenated_torch)
    passing = entire_tensor_passing and passing

    print(output)
    assert passing


@pytest.mark.parametrize("seq_len", [1024], ids=["seq_len_1024"])
@pytest.mark.parametrize("num_cores", [64])
def test_v1_sharded_sequence_fused_softmax(
    device,
    seq_len,
    num_cores,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    grid_size = (8, 8)
    num_heads = 64

    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16
    elif seq_len == 128:
        num_slices = 1

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]
    attention_mask_proper_dim_shape = [1, 1, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask_proper_dim = torch.randn(attention_mask_proper_dim_shape).bfloat16().float()
    torch_attention_mask = torch_attention_mask_proper_dim.repeat(1, attention_mask_shape[1], 1, 1)
    scalar_value = 1 / math.sqrt(num_heads)
    torch_scalar = (torch.ones(scalar_shape) * scalar_value).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )

    attention_mask_proper_dim = torch2tt_tensor(
        torch_attention_mask_proper_dim,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT4_B,
    )

    compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # We need to create attention masks per slice
    attention_masks_per_slice = []
    attention_mask_starting_index_per_slice = 0
    slice_length = (71 * seq_len) // num_slices
    number_of_attention_mask_elements_used_per_slice = slice_length - seq_len * (slice_length // seq_len)
    # print("Slice length is: ", slice_length)
    # print("Number of attention mask elements per slice = ", number_of_attention_mask_elements_used_per_slice)
    for slice_index in range(num_slices):
        print("Slice attention mask starting index: ", attention_mask_starting_index_per_slice)
        torch_attention_mask_per_slice = torch.cat(
            [
                torch_attention_mask_proper_dim[:, :, attention_mask_starting_index_per_slice:, :],
                torch_attention_mask_proper_dim[:, :, :attention_mask_starting_index_per_slice, :],
            ],
            dim=2,
        )
        tt_attention_slice = torch2tt_tensor(
            torch_attention_mask_per_slice,
            device,
            tt_memory_config=dram_interleaved_memory_config,
            tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT4_B,
        )
        attention_masks_per_slice.append(tt_attention_slice)
        attention_mask_starting_index_per_slice = (
            attention_mask_starting_index_per_slice + number_of_attention_mask_elements_used_per_slice
        ) % seq_len  # mod attention_mask.height

    reference_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )

    passing = True
    output = None

    attention_output_concatenated = torch2tt_tensor(
        torch_attention_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for i in range(num_slices):
        slice = ttnn.experimental.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8  # best option
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

        mm_slice = ttnn.matmul(
            slice,
            reference_key_layer_transposed,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        # Deallocating here causes pcc to drop - issue #6638
        # So we have to move it after the entire sequence is finished
        # slice.deallocate()

        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            mm_slice,
            scalar_value,
            attention_masks_per_slice[i],
            program_config=softmax_program_config,
            compute_kernel_config=compute_kernel_config,
        )

        subblock_w = 2
        subblock_h = 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

        attn_out_slice = ttnn.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
            compute_kernel_config=compute_kernel_config,
        )

        ttnn.experimental.tensor.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_concatenated,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        slice.deallocate()
        mm_slice.deallocate()
        attn_out_slice.deallocate()

    attention_output_concatenated_torch = tt2torch_tensor(attention_output_concatenated)

    print(output)
    assert passing


@pytest.mark.parametrize("seq_len", [1024], ids=["seq_len_1024"])
@pytest.mark.parametrize("num_cores", [64])
def test_baseline(
    device,
    seq_len,
    num_cores,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    num_heads = 64

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()
    torch_scalar = (torch.ones(scalar_shape) * (1 / math.sqrt(num_heads))).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.experimental.tensor.BufferType.DRAM,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16,
    )

    compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    attn_weights = ttnn.matmul(
        reference_query_layer, reference_key_layer_transposed, memory_config=dram_interleaved_memory_config
    )
    attn_weights = ttnn.experimental.operations.primary.bcast(
        attn_weights,
        reference_scalar,
        ttnn.experimental.tensor.BcastOpMath.MUL,
        ttnn.experimental.tensor.BcastOpDim.HW,
        output_mem_config=dram_interleaved_memory_config,
    )
    attn_weights = ttnn.add(attn_weights, attention_mask, memory_config=dram_interleaved_memory_config)
    attn_weights = ttnn.softmax_in_place(attn_weights, compute_kernel_config=compute_kernel_config)
    attn_output = ttnn.matmul(attn_weights, reference_value_layer)
    attn_output_torch = tt2torch_tensor(attn_output)

    passing = True
    assert passing
