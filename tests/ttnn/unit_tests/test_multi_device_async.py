# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor

#######
# Multi-Device Tensor tests running in async mode
#######


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_ttnn_to_and_from_multi_device_shard(pcie_device_mesh, layout, memory_config, dtype):
    """Shard a tensor across devices, compose it back and verify loopback tensor is same as the original tensor"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    for i in range(100):
        torch_tensor = torch.rand((1, 1, 256, 512), dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3)
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        torch_loop_back_tensor = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
        )
        assert_with_pcc(torch_tensor, torch_loop_back_tensor, pcc=0.9999)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_multi_device_check_per_device_shard(pcie_device_mesh, layout, memory_config, dtype):
    """This test checks if the tensor is correctly sharded across devices"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Unsupported test permutation: bfloat8_b with ROW_MAJOR_LAYOUT")

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    num_loops = 50
    if dtype == ttnn.bfloat8_b:
        # On host bfloat8_b conversion is slow. Decrease num loops.
        num_loops = 10
    for i in range(num_loops):
        torch_tensor = torch.rand((8, 1, 1024, 1024), dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3)
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh, memory_config=memory_config)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)

        shard_offset, shard_size = 0, int(1024 / len(pcie_device_mesh.get_device_ids()))
        for device_tensor in ttnn.get_device_tensors(ttnn_loop_back_tensor):
            device_tensor_torch = ttnn.to_torch(device_tensor)
            assert_with_pcc(
                device_tensor_torch, torch_tensor[..., shard_offset : shard_offset + shard_size], pcc=0.9999
            )
            shard_offset += shard_size

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 1040, 1040)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_multi_device_replicate(pcie_device_mesh, shape, layout, memory_config):
    """Test ReplicateTensorToMesh to broadcast a tensor across multiple devices"""
    from ttnn import ReplicateTensorToMesh, ListMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    for i in range(100):
        full_tensor = torch.rand(shape, dtype=torch.bfloat16)

        ttnn_tensor = ttnn.from_torch(
            full_tensor,
            mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
            layout=layout,
            memory_config=memory_config,
            device=pcie_device_mesh,
        )
        ttnn_tensor = ttnn.to_device(ttnn_tensor, pcie_device_mesh)
        ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
        loopback_replicated_tensors = ttnn.to_torch(
            ttnn_loop_back_tensor, mesh_composer=ListMeshToTensor(pcie_device_mesh)
        )
        for loopback_replicated_tensor in loopback_replicated_tensors:
            assert torch.all(full_tensor == loopback_replicated_tensor)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_ttnn_to_multi_device_tilized_parallel(pcie_device_mesh, layout, memory_config, dtype):
    """Test multi chip layout conversions on worker threads"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ListMeshToTensor

    shard_dim = 3
    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)
    for loop in range(20):
        torch_tensor = torch.rand((8, 1, 1024, 1024), dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=shard_dim),
            layout=layout,
            memory_config=memory_config,
            device=pcie_device_mesh,
        )
        if loop < 10:
            # Test Concat Composer
            readback_tensor = ttnn.to_torch(
                ttnn_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=shard_dim), device=pcie_device_mesh
            )
        else:
            # Test Mesh Composer
            readback_tensors = ttnn.to_torch(
                ttnn_tensor, mesh_composer=ListMeshToTensor(pcie_device_mesh), device=pcie_device_mesh
            )
            readback_tensor = torch.cat(readback_tensors, dim=shard_dim)
        assert torch.all(readback_tensor == torch_tensor)
    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("program_cache", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 3, 1024, 1024)])
def test_multi_device_unary_binary_op_chain(pcie_device_mesh, program_cache, shape):
    """Multidevice API test: Running tensor-parallel multi-device chain of eltwise ops"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)
        if program_cache:
            pcie_device_mesh.get_device(device).enable_program_cache()

    torch_silu = torch.nn.SiLU()
    for i in range(50):
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_output_golden = torch.add(
            torch.subtract(
                torch.exp(torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_tensor))),
                torch.exp(torch_input_tensor),
            ),
            torch_silu(torch_input_tensor),
        )

        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=3),
            device=pcie_device_mesh,
        )
        ttnn_output_tensor = ttnn.add(
            ttnn.sub(ttnn.exp(ttnn.relu(ttnn.gelu(ttnn_input_tensor))), ttnn.exp(ttnn_input_tensor)),
            ttnn.silu(ttnn_input_tensor),
        )
        ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor)
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=3)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.98)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("program_cache", [False, True])
@pytest.mark.parametrize("input_a_shape", [(4, 1, 512, 512), (16, 1, 512, 512)])
def test_multi_device_data_parallel_op_chain(pcie_device_mesh, program_cache, input_a_shape):
    """Multidevice API: Running data-parallel chain of ops with matmul"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)
        if program_cache:
            pcie_device_mesh.get_device(device).enable_program_cache()

    torch_silu = torch.nn.SiLU()
    torch_mish = torch.nn.Mish()
    for i in range(5):
        torch_input_a_tensor = torch.rand(input_a_shape, dtype=torch.bfloat16)
        torch_input_b_tensor = torch.rand((1, 1, 512, 512), dtype=torch.bfloat16)
        torch_output_golden = torch_mish(
            torch_silu(
                torch.nn.functional.relu(torch.nn.functional.gelu(torch_input_a_tensor @ torch_input_b_tensor))
                @ torch.exp(torch_input_a_tensor)
            )
        )

        ttnn_input_a_tensor = ttnn.from_torch(
            torch_input_a_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_device_mesh,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
        )
        ttnn_input_b_tensor = ttnn.from_torch(
            torch_input_b_tensor,
            layout=ttnn.TILE_LAYOUT,
            device=pcie_device_mesh,
            mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
        )
        ttnn_output_tensor = ttnn.from_device(
            ttnn.mish(
                ttnn.silu(
                    ttnn.relu(ttnn.gelu(ttnn_input_a_tensor @ ttnn_input_b_tensor)) @ ttnn.exp(ttnn_input_a_tensor)
                )
            )
        )
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.97)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("pcie_device_mesh", [2], indirect=True)
def test_multi_device_explicit_dealloc(pcie_device_mesh):
    """Multidevice API: Ensure that deallocating multi-device tensors works as expected"""
    from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    # Create input tensors that cause OOM during op execution
    # Explictly deallocate buffers after each op to ensure we don't run OOM.
    torch_input_a_tensor = torch.rand((512, 1, 2048, 2048), dtype=torch.bfloat16)
    torch_input_b_tensor = torch.rand((1, 1, 2048, 2048), dtype=torch.bfloat16)

    ttnn_input_a_tensor = ttnn.from_torch(
        torch_input_a_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        torch_input_b_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
    )
    ttnn_output_tensor_1 = ttnn_input_a_tensor @ ttnn_input_b_tensor
    ttnn_output_tensor_2 = ttnn.gelu(ttnn_output_tensor_1)
    ttnn_output_tensor_1.deallocate()
    ttnn_input_b_tensor.deallocate()
    ttnn_output_tensor_3 = ttnn.relu(ttnn_output_tensor_2)
    ttnn_output_tensor_2.deallocate()
    ttnn_output_tensor_4 = ttnn_output_tensor_3 @ ttnn_input_a_tensor
    ttnn_output_tensor_3.deallocate()
    ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor_4)
    ttnn_torch_output_tensor = ttnn.to_torch(
        ttnn_output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
    )

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("pcie_device_mesh", [2], indirect=True)
def test_add_1D_tensor_and_scalar(pcie_device_mesh, scalar, size):
    torch.manual_seed(0)

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(True)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=pcie_device_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(pcie_device_mesh),
    )
    output_tensor = input_tensor + scalar
    output_tensors = ttnn.to_torch(output_tensor, mesh_composer=ttnn.ListMeshToTensor(pcie_device_mesh))
    for output_tensor in output_tensors:
        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
        assert output_tensor.shape == (1, size)


# Async mode trace test: hang with all-gather enabled
# Event ID assert without all-gather. This is with pcie mapped devices.
def test_multi_device_multi_op_trace(pcie_device_mesh):
    # Trace requires program cache to be enabled
    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(True)
        pcie_device_mesh.get_device(device_id).enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 512, 512)), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh
    )
    input_1_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 512, 512)), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh
    )
    weight_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 512, 512)), ttnn.bfloat16, ttnn.TILE_LAYOUT, pcie_device_mesh
    )

    # Op chain to be traced
    def run_op_chain(input_0, input_1, weight):
        return ttnn.all_gather(
            ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1))) @ ttnn.silu(weight),
            dim=0,
            num_links=1,
        )

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev, weight_dev)
    # Capture Trace
    tid = ttnn.begin_multi_device_trace_capture(pcie_device_mesh, 53248, 0)
    # Output tensor will be reused across trace runs, since its buffer will be updated with trace output
    output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_multi_device_trace_capture(pcie_device_mesh, tid, 0)

    # Execute and verify trace against pytorch
    torch_silu = torch.nn.SiLU()
    for i in range(25):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand((pcie_device_mesh.get_num_devices(), 1, 512, 512), dtype=torch.bfloat16)
        torch_input_tensor_1 = torch.rand((pcie_device_mesh.get_num_devices(), 1, 512, 512), dtype=torch.bfloat16)
        torch_weight = torch.rand((1, 1, 512, 512), dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        ) @ torch_silu(torch_weight)

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0)
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, layout=ttnn.TILE_LAYOUT, mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh)
        )
        # Copy TTNN host tensors into preallocated Mult-Device tensors
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)
        ttnn.copy_host_to_device_tensor(ttnn_weight, weight_dev)
        # Execute trace (non-blocking through CQ 0): sends output to output_tensor
        # output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
        ttnn.execute_multi_device_trace(pcie_device_mesh, tid, 0, False)
        # Compare
        # ttnn_torch_output_tensor = ttnn.to_torch(
        #     output_tensor, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0)
        # )
        # print("Readback done")
        # passed, msg = assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.8)
        # logger.info(msg)
        device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(output_tensor)
        for device_tensor in device_tensors:
            device_tensor_torch = ttnn.to_torch(device_tensor)
            passed, msg = assert_with_pcc(device_tensor_torch, torch_output_golden, pcc=0.99)
            logger.info(msg)

    # Release trace buffer once workload is complete
    ttnn.release_multi_device_trace(pcie_device_mesh, tid)

    for device_id in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device_id).enable_async(False)
