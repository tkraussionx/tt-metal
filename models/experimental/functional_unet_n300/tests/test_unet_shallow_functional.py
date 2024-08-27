# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from loguru import logger

from models.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_grayskull,
    divup,
    profiler,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.experimental.functional_unet_n300.unet_utils import create_unet_models, create_unet_input_tensors

import tt_lib as ttl

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def setup_l1_sharded_input(device, tt_inputs, tt_model, mesh_mapper, mesh_composer):
    num_devices = 1 if isinstance(device, ttnn.Device) else device.get_num_devices()

    padded_input_shape, input_mem_config, _ = ttnn.get_conv_padded_input_shape_and_mem_config(
        device=device,
        input_tensor=tt_inputs,
        conv_config=tt_model.conv1_config,
        batch_size=tt_model.batch_size,
        height=tt_model.conv1_output_height,
        width=tt_model.conv1_output_width,
        in_channels=tt_model.conv1_input_channels,
        out_channels=tt_model.conv1_output_channels,
    )
    print(input_mem_config)
    inputs_padded = ttnn.to_torch(tt_inputs, device=device, mesh_composer=mesh_composer)
    inputs_padded = inputs_padded.reshape(num_devices, 1, -1, inputs_padded.shape[-1])
    inputs_padded = torch.nn.functional.pad(
        inputs_padded,
        (0, padded_input_shape[-1] - inputs_padded.shape[-1], 0, padded_input_shape[-2] - inputs_padded.shape[-2]),
    )
    tt_inputs_host = ttnn.from_torch(
        inputs_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
    )
    return tt_inputs_host, input_mem_config


def setup_dram_sharded_input(device, tt_inputs, tt_model, mesh_mapper, mesh_composer):
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(device, tt_inputs, tt_model, mesh_mapper, mesh_composer)
    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
            tt_inputs_host.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config


def run_trace_2cq_model(
    device,
    tt_inputs,
    tt_model,
    mesh_mapper,
    mesh_composer,
    num_warmup_iterations,
    num_measurement_iterations,
    original_shape,
):
    ops_parallel_config = {}
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = setup_dram_sharded_input(
        device, tt_inputs, tt_model, mesh_mapper, mesh_composer
    )
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)

    op_event = ttnn.create_event(device)
    write_event = ttnn.create_event(device)
    # Initialize the op event so we can write
    ttnn.record_event(0, op_event)

    profiler.start("compile")
    print("start")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    print("reshard_out")
    reshard_out = ttnn.to_memory_config(tt_image_res, input_mem_config)
    print("reshard_outreshard_out")
    ttnn.record_event(0, op_event)
    perf_mode = False
    print("tt_model")
    _ = tt_model(device, reshard_out, original_shape, perf_mode, ops_parallel_config)
    profiler.end("compile")
    print("compile")
    ttnn.dump_device_profiler(device)

    profiler.start("cache")
    print("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)
    ttnn.wait_for_event(0, write_event)
    reshard_out = ttnn.to_memory_config(tt_image_res, input_mem_config)
    first_out_addr = ttnn.buffer_address(reshard_out)
    ttnn.record_event(0, op_event)
    _ = tt_model(device, reshard_out, original_shape, perf_mode, ops_parallel_config)
    profiler.end("cache")
    print("cache")
    ttnn.dump_device_profiler(device)

    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    ttnn.record_event(1, write_event)

    ttnn.wait_for_event(0, write_event)
    reshard_out = ttnn.to_memory_config(tt_image_res, input_mem_config)
    ttnn.record_event(0, op_event)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = tt_model(device, reshard_out, original_shape, perf_mode, ops_parallel_config)
    reshard_out = ttnn.allocate_tensor_on_device(
        reshard_out.shape, reshard_out.dtype, reshard_out.layout, device, input_mem_config
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert first_out_addr == ttnn.buffer_address(reshard_out)
    ttnn.dump_device_profiler(device)

    for iter in range(0, num_warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        reshard_out = ttnn.reshard(tt_image_res, input_mem_config, reshard_out)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.dump_device_profiler(device)

    ttnn.synchronize_devices(device)
    if use_signpost:
        signpost(header="start")
    outputs = []
    profiler.start(f"run")
    for iter in range(0, num_measurement_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        ttnn.record_event(1, write_event)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        reshard_out = ttnn.reshard(tt_image_res, input_mem_config, reshard_out)
        ttnn.record_event(0, op_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(tt_output_res.cpu(blocking=False))
    ttnn.synchronize_devices(device)
    profiler.end(f"run")
    print(f"run")
    if use_signpost:
        signpost(header="stop")
    ttnn.dump_device_profiler(device)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("mode", ["2cq_with_trace"])
def test_unet_pcc(device_mesh, perf_mode, batch, groups, mode):
    with torch.no_grad():
        torch.manual_seed(0)
        print("enter")
        num_devices = 1 if isinstance(device_mesh, ttnn.Device) else device_mesh.get_num_devices()

        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)
        print("mesh")

        # Create initial parameters
        torch_input_tensor_tt, ttnn_input_tensor = create_unet_input_tensors(
            device_mesh, batch, groups, inputs_mesh_mapper
        )
        print("input")
        torch_model, ttnn_model = create_unet_models(
            device_mesh,
            batch,
            groups,
            torch_input_tensor_tt,
            weights_mesh_mapper=weights_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )
        print("torch")
        torch_input_tensor = torch.randn(num_devices * batch, 4 * groups, 1056, 160)
        # Run torch golden result
        torch_output_tensor = torch_model(torch_input_tensor)
        if mode == "default":
            print("default")
            # Run ttnn output result
            output_tensor = ttnn_model(
                device_mesh, ttnn_input_tensor, list(torch_input_tensor_tt.shape), perf_mode=perf_mode
            )
            # Tensor postprocessing
            output_tensor = ttnn.to_torch(output_tensor, device=device_mesh, mesh_composer=output_mesh_composer)
            output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
            output_tensor = torch.reshape(
                output_tensor,
                [
                    num_devices * batch,
                    1,
                    1058,
                    162,
                ],
            )
            output_tensor = output_tensor[:, :, 1:-1, 1:-1]
            output_tensor = output_tensor.to(torch_input_tensor.dtype)
            assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
        elif mode == "2cq_with_trace":
            print("2cq_with_trace")
            num_warmup_iterations = 5
            num_measurement_iterations = 15
            run_trace_2cq_model(
                device_mesh,
                ttnn_input_tensor,
                ttnn_model,
                inputs_mesh_mapper,
                output_mesh_composer,
                num_warmup_iterations,
                num_measurement_iterations,
                list(torch_input_tensor_tt.shape),
            )
            print("finished")
