# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor, pad_by_zero, get_devices_for_t3000


@pytest.mark.parametrize(
    "shard_orientation",
    (ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR),
)
@pytest.mark.parametrize(
    "output_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in1_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in0_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads, kv_heads",
    (
        (32, 64, 512 + 96, 16, 1),  # 8 chip pre-attn matmul shapes
        (32, 1024 + 32, 64, 16, 1),  # 8 chip post-attn matmul shapes
    ),
)
def test_group_attn_matmul(
    batch, K, seq_len, q_heads, kv_heads, in0_sharded, in1_sharded, output_sharded, shard_orientation, device
):
    torch.manual_seed(0)

    compute_grid_size = device.compute_with_storage_grid_size()

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    # NOTE: Mixed precision is supported as well; but might not have enough space for larger seq_len with BFLOAT16
    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    output_dtype = ttl.tensor.DataType.BFLOAT8_B

    q_len = 1
    input_shape_a = [q_len, q_heads, batch, K]
    input_shape_b = [batch, kv_heads, K, seq_len]

    input_tensor_a = torch.randn(input_shape_a).bfloat16()
    input_tensor_b = torch.randn(input_shape_b).bfloat16()

    tt_input_tensor_a = (
        ttl.tensor.Tensor(input_tensor_a, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, interleaved_mem_config)
    )
    tt_input_tensor_b = (
        ttl.tensor.Tensor(input_tensor_b, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, interleaved_mem_config)
    )

    if in0_sharded:
        tt_input_tensor_a = ttl.tensor.interleaved_to_sharded(
            tt_input_tensor_a,
            compute_grid_size,
            [q_len * batch, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if in1_sharded:
        tt_input_tensor_b = ttl.tensor.interleaved_to_sharded(
            tt_input_tensor_b,
            compute_grid_size,
            [kv_heads * K, seq_len],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if output_sharded:
        output_mem_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        )
    else:
        output_mem_config = interleaved_mem_config

    tt_output_tensor_on_device = ttl.operations.primary.transformers.group_attn_matmul(
        tt_input_tensor_a,
        tt_input_tensor_b,
        compute_with_storage_grid_size=compute_grid_size,
        output_mem_config=output_mem_config,
        output_dtype=output_dtype,
    )
    if output_sharded:
        tt_output_tensor_on_device = ttl.tensor.sharded_to_interleaved(
            tt_output_tensor_on_device, interleaved_mem_config
        )

    tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    input_tensor_a = input_tensor_a.to(torch.float)
    input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
    golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
    assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize(
    "M, K, N, num_cores",
    [
        [32, 8192, 1152, 8],
    ],
)
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT16])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in0(
    device, in0_sharded, out_sharded, M, K, N, num_cores, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = (8, 1)

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=32,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize(
    "M, K, N, num_cores",
    [
        [32, 8192, 65024, 32],
    ],
    ids=["lm_head_shape"],
)
@pytest.mark.parametrize("out_sharded", [False], ids=["out_sharded"])
@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("weights_dtype", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("activations_dtype", [ttl.tensor.DataType.BFLOAT8_B])
def test_sharded_matmul_1d_in0_multi_chip(
    all_devices,
    num_devices,
    use_program_cache,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    num_cores,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    # if num_devices == 8:
    #     pytest.skip("Need tunnelling support to run on 8 devices!")

    grid_size = (8, 4)
    devices = get_devices_for_t3000(all_devices, num_devices)
    print("Running with " + str(all_devices))
    print("Num Devices: " + str(num_devices))

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    l1_interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in1_slices = torch.chunk(in1, num_devices, dim=-1)
    for i in range(2):
        in0_t = []
        in1_t = []
        for i in range(num_devices):
            logger.info(f"Putting tensors on device: {i}")
            in0_temp = torch2tt_tensor(
                in0, devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
            )

            if in0_sharded:
                in0_temp = ttl.tensor.interleaved_to_sharded(
                    in0_temp,
                    grid_size,
                    [M, K // num_cores],
                    ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                )
            in0_t.append(in0_temp)

            in1_t.append(
                torch2tt_tensor(
                    in1_slices[i], devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype
                )
            )

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

        if num_devices == 4:
            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=16,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        elif num_devices == 8:
            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        output_t = []
        for i in range(num_devices):
            logger.info(f"Running matmul on device: {i}")
            output_t.append(
                ttl.operations.primary.matmul_1d(
                    in0_t[i],
                    in1_t[i],
                    program_config=program_config,
                    output_mem_config=l1_interleaved_mem_config,
                    output_dtype=activations_dtype,
                )
            )

        pt_out = in0 @ in1

        # tt_out = torch.cat([tt2torch_tensor(out_t) for out_t in output_t], -1)
        tt_out = tt2torch_tensor(ttl.tensor.all_gather(output_t, 3, 1, l1_interleaved_mem_config)[0])
        passing, output = comp_pcc(pt_out, tt_out)
        logger.info(output)
        assert passing
