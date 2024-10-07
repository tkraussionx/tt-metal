# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from ttnn import experimental as ttl
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from models.utility_functions import nearest_32
import math

shard_spec_24_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 2),
        ),
    }
)

TILE_SIZE = 32
shard_height = TILE_SIZE  # Decode mode only
cluster_size = (4, 8)
hidden_size = 8192

hidden_size_24_pad = (hidden_size + (cluster_size[0] * 24 * TILE_SIZE // 3)) // cluster_size[0]
shard_width_hidden_dim_across_24_cores = hidden_size_24_pad // 24


def run_prefetch_matmul_on_t3000_impl(
    t3k_mesh_device,
    input_shape,
    input_dtype,
    layout,
    # Matmul params
    N,
    weight_shard_dim,
    core_grid,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    math_fidelity,
    # Memory configs
    mem_config_input,
    mem_config_weights,
    mem_config_mm,
    num_iters=1,
    enable_trace=False,
):
    devices = t3k_mesh_device.get_devices()
    num_devices = len(devices)

    ##### Create input tensor for the all gather #####
    _, _, M, K = input_shape
    input_tensor = torch.randn(input_shape).float()
    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=input_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_input,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )
    logger.info(f"Input tensor shape: {tt_input_tensor.shape}")

    ##### Config for the weight matrix #####
    if mem_config_weights == "dram":
        mem_config_weights = ttnn.DRAM_MEMORY_CONFIG
    elif mem_config_weights == "l1":
        mem_config_weights = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_24_cores_grid,
                [
                    K,
                    N // mem_config_input.shard_spec.num_cores(),
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    else:
        raise ValueError(f"Unsupported mem_config_weights: {mem_config_weights}")

    ##### Create the weight matrix for the matmul #####
    weights_tensor = torch.randn([1, 1, K, N * num_devices]).float()
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=weight_shard_dim),
    )
    logger.info(f"Weight tensor shape: {weight_tt.shape}")

    ##### Configs for ttnn.matmul #####
    if matmul_config == "matmul_1d_ff1":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=max(1, math.ceil(K / 32)),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=5,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=shard_height // 32,  # M / TILE_HEIGHT / Grid_Size
            per_core_N=max(1, math.ceil(N / 32 / (core_grid[0] * core_grid[1]))),  # N / TILE_WIDTH / Grid_Size
            mcast_in0=True,
            gather_in0=True,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=True,
        )
    else:
        raise ValueError(f"Unsupported matmul_config: {matmul_config}")

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ##### Perform the torch ops #####
    matmul_output = torch.matmul(input_tensor, weights_tensor)

    ##### Perform the TT ops #####
    def run_op():
        if core_grid is None or isinstance(core_grid, tuple):
            return ttnn.matmul(
                tt_input_tensor,
                weight_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            return ttnn.matmul(
                tt_input_tensor,
                weight_tt,
                core_grid=core_grid,
                memory_config=mem_config_mm,
                compute_kernel_config=compute_kernel_config,
            )

    if enable_trace:
        # Compile the op
        tt_matmul_out_tensor = run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_matmul_out_tensor = run_op()
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)

        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        for d in devices:
            ttnn.synchronize_device(d)
    else:
        for i in range(num_iters):
            tt_matmul_out_tensor = run_op()

            # Synchronize the devices
            for d in devices:
                ttnn.synchronize_device(d)

            logger.info(f"Done iteration {i}")

    print("Checking outputs for Matmul")
    tt_mm_out = ttnn.from_device(tt_matmul_out_tensor)
    tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=weight_shard_dim))

    eq, output = comp_pcc(tt_mm_out, matmul_output)
    logger.info(f"Output: {output}")
    assert eq


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype, matmul_weights_dtype, math_fidelity",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    ],
)
@pytest.mark.parametrize(
    "matmul_config, input_shape, N, weight_shard_dim, core_grid, max_in0_block_w, mem_config_input, mem_config_weights, mem_config_mm",
    [
        (  # FF1/3 matmul1d
            "matmul_1d_ff1",
            [1, 1, 32, hidden_size_24_pad],
            3584 + 256,
            3,
            (8, 3),
            4,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_24_cores_grid,
                    [
                        shard_height,
                        shard_width_hidden_dim_across_24_cores,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            "l1",
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ),
    ],
    ids=[
        "ff1_matmul1d_24",
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 90112}], indirect=True)
def test_prefetch_matmul_on_t3000(
    t3k_mesh_device,
    input_shape,
    input_dtype,
    layout,
    N,
    weight_shard_dim,
    core_grid,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    math_fidelity,
    mem_config_input,
    mem_config_weights,
    mem_config_mm,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_prefetch_matmul_on_t3000_impl(
        t3k_mesh_device,
        input_shape,
        input_dtype,
        layout,
        N,
        weight_shard_dim,
        core_grid,
        matmul_config,
        matmul_weights_dtype,
        max_in0_block_w,
        math_fidelity,
        mem_config_input,
        mem_config_weights,
        mem_config_mm,
    )
