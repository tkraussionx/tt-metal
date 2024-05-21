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
import time


@pytest.mark.parametrize("seq_len", [1024], ids=["seq_len_1024"])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_devices", [1, 8], ids=["1chips", "8chips"])
def test_min_repro(
    all_devices,
    seq_len,
    num_cores,
    num_devices,
    function_level_defaults,
):
    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices.append(all_devices[0])

    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    print("Running with: ", num_devices, " devices.")

    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16
    elif seq_len == 128:
        num_slices = 1

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer_per_device = []
    reference_key_layer_transposed_per_device = []
    for device_idx in range(num_devices):
        reference_query_layer_per_device.append(
            torch2tt_tensor(
                torch_query_layer,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )
        reference_key_layer_transposed_per_device.append(
            torch2tt_tensor(
                torch_key_layer_transposed,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    reference_value_layer_per_device = []
    attention_output_concatenated_per_device = []
    for device_idx in range(num_devices):
        reference_value_layer_per_device.append(
            torch2tt_tensor(
                torch_value_layer,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )
        attention_output_concatenated_per_device.append(
            torch2tt_tensor(
                torch_attention_output,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )

    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    print("Tiles per shard is: ", tiles_per_shard)

    slices = []
    for device_idx in range(num_devices):
        slices.append(
            ttl.tensor.interleaved_to_sharded_partial(
                reference_query_layer_per_device[device_idx],
                grid_size,
                mm_activations_height_shard_spec,
                num_slices,  # num_slices
                0,  # slice_index
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )
        )

    print("Sleeping for 20 seconds after I2SP")
    time.sleep(20)

    print("I2SP -> Begin sync")
    for device_idx in range(num_devices):
        ttl.device.Synchronize(devices[device_idx])
    print("I2SP -> End sync")

    print("Sleeping for 30 seconds after Sync point")
    print(
        "MM dims: IN0",
        slices[device_idx].get_legacy_shape(),
        " IN1: ",
        reference_key_layer_transposed_per_device[device_idx].get_legacy_shape(),
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        per_core_M=tiles_per_shard,
        per_core_N=seq_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    print("MM1 program config: ", program_config)

    # mm_slices = []
    # for device_idx in range(num_devices):
    #     mm_slices.append(
    #         ttl.operations.primary.matmul(
    #             slices[device_idx],
    #             reference_key_layer_transposed_per_device[device_idx],
    #             program_config=program_config,
    #             output_mem_config=height_sharded_memory_config,
    #             output_dtype=ttl.tensor.DataType.BFLOAT16,
    #             compute_kernel_config=compute_kernel_config,
    #         )
    #     )

    # print("Sleeping for 20 seconds after MM1")
    # time.sleep(20)

    # print("MM1 -> Begin sync")
    # for device_idx in range(num_devices):
    #     ttl.device.Synchronize(devices[device_idx])
    # print("MM1 -> End sync")

    # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    #     compute_with_storage_grid_size=grid_size,
    #     in0_block_w=seq_len // 32,
    #     per_core_M=tiles_per_shard,
    #     per_core_N=2,
    #     out_subblock_h=1,
    #     out_subblock_w=1,
    #     fuse_batch=True,
    #     fused_activation=None,
    #     mcast_in0=False,
    # )

    # attn_out_slices = []
    # for device_idx in range(num_devices):
    #     attn_out_slices.append(
    #         ttl.operations.primary.matmul(
    #             mm_slices[device_idx],
    #             reference_value_layer_per_device[device_idx],
    #             program_config=program_config,
    #             output_mem_config=height_sharded_memory_config,
    #             output_dtype=ttl.tensor.DataType.BFLOAT16,
    #             compute_kernel_config=compute_kernel_config,
    #         )
    #     )

    # print("Sleeping for 20 seconds after MM2")
    # time.sleep(20)
    # print("MM2 -> Begin sync")
    # for device_idx in range(num_devices):
    #     ttl.device.Synchronize(devices[device_idx])
    # print("MM2 -> End sync")

    # for device_idx in range(num_devices):
    #     ttl.tensor.sharded_to_interleaved_partial(
    #         attn_out_slices[device_idx],
    #         attention_output_concatenated_per_device[device_idx],
    #         num_slices,
    #         0,
    #         dram_interleaved_memory_config,
    #     )

    # print("Sleeping for 20 seconds after S2IP")
    # time.sleep(20)
    # print("S2IP -> Begin sync")
    # for device_idx in range(num_devices):
    #     ttl.device.Synchronize(devices[device_idx])
    # print("S2IP -> End sync")

    for device_idx in range(num_devices):
        slices[device_idx].deallocate()
        # mm_slices[device_idx].deallocate()
        # attn_out_slices[device_idx].deallocate()

    print("Done!")
    passing = True
    assert passing


@pytest.mark.parametrize("seq_len", [1024, 2048], ids=["seq_len_1024", "seq_len_2048"])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_devices", [1, 8], ids=["1chips", "8chips"])
@pytest.mark.parametrize("loops", [1, 10, 100, 1000], ids=["1_loops", "10_loops", "100_loops", "1000_loops"])
def test_falcon7b_crash(
    all_devices,
    seq_len,
    num_cores,
    num_devices,
    loops,
    use_program_cache,
    function_level_defaults,
):
    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices.append(all_devices[0])

    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    num_heads = 64
    print("Running with: ", num_devices, " devices and loops: ", loops)

    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16
    elif seq_len == 128:
        num_slices = 1

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_proper_dim_shape = [1, 1, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask_proper_dim = torch.randn(attention_mask_proper_dim_shape).bfloat16().float()
    scalar_value = 1 / math.sqrt(num_heads)
    torch_scalar = (torch.ones(scalar_shape) * scalar_value).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer_per_device = []
    reference_key_layer_transposed_per_device = []
    for device_idx in range(num_devices):
        reference_query_layer_per_device.append(
            torch2tt_tensor(
                torch_query_layer,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )
        reference_key_layer_transposed_per_device.append(
            torch2tt_tensor(
                torch_key_layer_transposed,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # We need to create attention masks per slice
    attention_masks_per_slice_per_device = []
    attention_mask_starting_index_per_slice = 0
    slice_length = (71 * seq_len) // num_slices
    number_of_attention_mask_elements_used_per_slice = slice_length - seq_len * (slice_length // seq_len)
    # print("Slice length is: ", slice_length)
    # print("Number of attention mask elements per slice = ", number_of_attention_mask_elements_used_per_slice)
    torch_attention_mask_per_slice = []
    for slice_index in range(num_slices):
        # print("Slice attention mask starting index: ", attention_mask_starting_index_per_slice)
        torch_attention_mask_per_slice.append(
            torch.cat(
                [
                    torch_attention_mask_proper_dim[:, :, attention_mask_starting_index_per_slice:, :],
                    torch_attention_mask_proper_dim[:, :, :attention_mask_starting_index_per_slice, :],
                ],
                dim=2,
            )
        )
        attention_mask_starting_index_per_slice = (
            attention_mask_starting_index_per_slice + number_of_attention_mask_elements_used_per_slice
        ) % seq_len  # mod attention_mask.height

    for device_idx in range(num_devices):
        attn_mask_slices = []
        for slice_index in range(num_slices):
            attn_mask_slices.append(
                torch2tt_tensor(
                    torch_attention_mask_per_slice[slice_index],
                    devices[device_idx],
                    tt_memory_config=dram_interleaved_memory_config,
                    tt_dtype=ttl.tensor.DataType.BFLOAT16,
                )
            )
        attention_masks_per_slice_per_device.append(attn_mask_slices)

    reference_scalar_per_device = []
    reference_value_layer_per_device = []
    attention_output_concatenated_per_device = []
    for device_idx in range(num_devices):
        reference_scalar_per_device.append(
            torch2tt_tensor(
                torch_scalar,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )
        reference_value_layer_per_device.append(
            torch2tt_tensor(
                torch_value_layer,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )
        attention_output_concatenated_per_device.append(
            torch2tt_tensor(
                torch_attention_output,
                devices[device_idx],
                tt_memory_config=dram_interleaved_memory_config,
                tt_dtype=ttl.tensor.DataType.BFLOAT16,
            )
        )

    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for l in range(loops):
        for i in range(num_slices):
            slices = []
            for device_idx in range(num_devices):
                slices.append(
                    ttl.tensor.interleaved_to_sharded_partial(
                        reference_query_layer_per_device[device_idx],
                        grid_size,
                        mm_activations_height_shard_spec,
                        num_slices,  # num_slices
                        i,  # slice_index
                        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttl.tensor.ShardOrientation.ROW_MAJOR,
                    )
                )

            subblock_h = 1
            subblock_w = 1
            if seq_len == 2048:
                subblock_w = 1  # best option
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

            mm_slices = []
            for device_idx in range(num_devices):
                mm_slices.append(
                    ttl.operations.primary.matmul(
                        slices[device_idx],
                        reference_key_layer_transposed_per_device[device_idx],
                        program_config=program_config,
                        output_mem_config=height_sharded_memory_config,
                        output_dtype=ttl.tensor.DataType.BFLOAT16,
                        compute_kernel_config=compute_kernel_config,
                    )
                )

            # Deallocating here causes pcc to drop - issue #6638
            # So we have to move it after the entire sequence is finished
            # slice.deallocate()

            subblock_w = 1
            if seq_len == 2048:
                subblock_w = 1
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=subblock_w,
                block_h=mm_output_height_shard_spec[0] // 32,
                block_w=mm_output_height_shard_spec[1] // 32,
            )

            for device_idx in range(num_devices):
                mm_slices[device_idx] = ttl.operations.primary.transformers.scale_causal_mask_hw_dims_softmax_in_place(
                    mm_slices[device_idx],
                    scalar_value,
                    attention_masks_per_slice_per_device[device_idx][i],
                    program_config=softmax_program_config,
                    compute_kernel_config=compute_kernel_config,
                )

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

            attn_out_slices = []
            for device_idx in range(num_devices):
                attn_out_slices.append(
                    ttl.operations.primary.matmul(
                        mm_slices[device_idx],
                        reference_value_layer_per_device[device_idx],
                        program_config=program_config,
                        output_mem_config=height_sharded_memory_config,
                        output_dtype=ttl.tensor.DataType.BFLOAT16,
                        compute_kernel_config=compute_kernel_config,
                    )
                )

            for device_idx in range(num_devices):
                ttl.tensor.sharded_to_interleaved_partial(
                    attn_out_slices[device_idx],
                    attention_output_concatenated_per_device[device_idx],
                    num_slices,
                    i,
                    dram_interleaved_memory_config,
                )

            for device_idx in range(num_devices):
                slices[device_idx].deallocate()
                mm_slices[device_idx].deallocate()
                attn_out_slices[device_idx].deallocate()

    print("Begin sync")
    for device_idx in range(num_devices):
        ttl.device.Synchronize(devices[device_idx])
    print("End sync")

    print("Done!")
    passing = True
    assert passing


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

    in0_tt = torch2tt_tensor(
        torch_in0,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    in1_tt = torch2tt_tensor(
        torch_in1,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    tiles_per_shard = 9

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

    # inline Tensor interleaved_to_sharded(
    #     const Tensor &input_tensor,
    #     const MemoryConfig &sharded_mem_config,
    #     std::optional<const DataType> output_dtype = std::nullopt)

    # inline Tensor interleaved_to_sharded(
    #     const Tensor &input_tensor,
    #     const std::variant<CoreCoord, CoreRangeSet> grid,
    #     const std::array<uint32_t, 2> shard_shape,
    #     const TensorMemoryLayout shard_scheme,
    #     const ShardOrientation shard_orientation,
    #     const std::optional<const DataType> output_dtype = std::nullopt

    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]

    in0_sharded = ttl.tensor.interleaved_to_sharded(
        in0_tt,
        grid_size,
        mm_activations_height_shard_spec,
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        ttl.tensor.DataType.BFLOAT16,
    )

    tt_out = ttl.operations.primary.matmul(
        in0_sharded,
        in1_tt,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
        compute_kernel_config=compute_kernel_config,
    )

    out = tt2torch_tensor(tt_out)
    passing = True
    assert passing
