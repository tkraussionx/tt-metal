# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from ttnn import experimental as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_all_gather import is_unsupported_case

ENABLE_TRACE = True
BLOCKING = True  # TODO: BLOCKING=True causes the test to hang


def run_all_gather_on_t3000_impl(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    # Memory configs
    mem_config_input,
    mem_config_ag,
    num_iters=1,
):
    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config_ag, num_devices, num_links, ag_input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = t3k_mesh_device.get_devices()

    logger.info(f"All Gather output shape: {ag_output_shape}")
    logger.info(f"dim: {dim}")

    ##### Create input tensor for the all gather #####
    _, _, _, hidden_dim = ag_output_shape
    input_tensor = torch.randn(ag_output_shape).float()
    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, ag_input_dtype).to(layout).to(devices[i], mem_config_input))
    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    ##### Perform the TT ops #####
    def run_op():
        # all_gather
        return ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config_ag)

    if ENABLE_TRACE:
        # Compile the op
        tt_all_gather_out_tensor = run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op()
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)

        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=BLOCKING)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        for d in devices:
            ttnn.synchronize_device(d)
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op()

            # Synchronize the devices
            for d in devices:
                ttnn.synchronize_device(d)

            logger.info(f"Done iteration {i}")

    ##### Compare the outputs #####
    print("Checking outputs for All Gather Matmul (All Gather)")
    for i, t in enumerate(ttnn.get_device_tensors(tt_all_gather_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if ag_input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        logger.info(f"Output {i}: {output}")
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"

    if ENABLE_TRACE:
        ttnn.release_trace(t3k_mesh_device, trace_id)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout",
    [
        (  # Llama decode Selfout
            8,
            1,
            [1, 1, 32, 1024 * 8],
            3,
            ttl.tensor.Layout.TILE,
        ),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(7, 0),
                            ),
                        }
                    ),
                    [
                        32,  # shard_height
                        128,  # shard width
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(7, 0),
                            ),
                        }
                    ),
                    [
                        32,  # shard_height
                        8192 // 8,  # shard_width_hidden_dim_across_8_cores
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        ),
    ],
    ids=(
        "non-sharded",
        "llama_selfout",
    ),
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112}], indirect=True
)  # TODO: Update once trace fails
def test_all_gather_matmul_1d_llama_selfout_on_t3000_post_commit(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_t3000_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
    )
