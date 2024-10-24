# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull


def run_normal(
    device,
    num_devices,
    input_shape,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
    enable_async,
):
    print("Running barrier test")
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    if enable_async:
        logger.info(f"Using Async Mode for Barrier Op Dispatch")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")
    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    # Use Async mode based on test input config
    device.enable_async(enable_async)
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(device.get_devices()[i], mem_config))
    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=mem_config,
            topology=all_gather_topology,
        )


def run_with_trace(
    device, all_gather_topology, input_tensor_mesh, dim, num_links, output_mem_config, n_worker, n_buffer, num_iter
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.all_gather(
        input_tensor_mesh,
        dim,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
        topology=all_gather_topology,
    )
    ttnn.barrier(
        input_tensor_mesh,
        memory_config=output_mem_config,
        topology=all_gather_topology,
    )
    for d in device.get_devices():
        ttnn.synchronize_device(d)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh,
            dim,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
            topology=all_gather_topology,
        )
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=output_mem_config,
            topology=all_gather_topology,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for d in device.get_devices():
        ttnn.synchronize_device(d)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    for d in device.get_devices():
        ttnn.synchronize_device(d)
    return tt_out_tensor


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        (8),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "input_shape, output_shard_spec,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
def test_run_barrier_impl(
    mesh_device,
    num_devices,
    input_shape,
    output_shard_spec,
    shard_grid,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
    enable_async,
):
    run_normal(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        input_dtype,
        mem_config,
        layout,
        num_iters,
        all_gather_topology,
        enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        (4),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "input_shape, output_shard_spec,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
def test_run_barrier_impl_pcie(
    pcie_mesh_device,
    num_devices,
    input_shape,
    output_shard_spec,
    shard_grid,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
    enable_async,
):
    run_normal(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        input_dtype,
        mem_config,
        layout,
        num_iters,
        all_gather_topology,
        enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("num_iter", [1000])
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 7840768}], indirect=True)
def test_barrier_sharded(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    use_program_cache,
    function_level_defaults,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    all_gather_topology,
    enable_async,
    trace_mode,
    num_iter,
):
    n_worker = None
    n_buffer = None
    t3k_mesh_device.enable_async(enable_async)
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    unchunked_input_tensor = torch.rand(unchunked_input_shape).bfloat16()

    unchunked_input_tensor = unchunked_input_tensor.bfloat16()

    input_tensors = torch.chunk(unchunked_input_tensor, num_devices, dim)

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
        False,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_shard_shape = list(input_shard_shape)
    if dim == 3:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
        False,
    )
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")

    tt_input_tensors = []

    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(
            ttnn.Tensor(t, input_dtype).to(tensor_layout).to(t3k_mesh_device.get_devices()[i], input_mem_config)
        )

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    if trace_mode:
        tt_out_tensor = run_with_trace(
            t3k_mesh_device,
            all_gather_topology,
            input_tensor_mesh,
            dim,
            num_links,
            output_mem_config,
            n_worker,
            n_buffer,
            num_iter,
        )
    else:
        ## Run the actual allgather operation
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                num_links=num_links,
                memory_config=output_mem_config,
                num_workers=n_worker,
                num_buffers_per_channel=n_buffer,
                topology=all_gather_topology,
            )
            ttnn.barrier(
                input_tensor_mesh,
                memory_config=output_mem_config,
                topology=all_gather_topology,
            )
        ## Wait for completion
        for d in t3k_mesh_device.get_devices():
            ttnn.synchronize_device(d)
