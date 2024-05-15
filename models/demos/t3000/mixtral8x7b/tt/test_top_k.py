import ttnn
import tt_lib
import torch
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor, ShardTensorToMesh


def test_top_2(device_mesh, use_program_cache):
    gate_logits_1SB8 = ttnn.from_torch(
        torch.randn((1, 1, 32, 64)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    expert_mask_11B2 = ttnn.from_torch(
        torch.cat([torch.full((1, 1, 32, 32), fill_value=1) for i in range(8)], dim=3),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
    )
    top2_mask = torch.full((1, 1, 32, 32), fill_value=torch.finfo(torch.float).min)
    top2_mask[:, :, :, :2] = 0.0
    top2_mask_11B_64 = ttnn.from_torch(
        top2_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    for i in range(10):
        start_time = time.time()
        ttl_topk_values, ttl_topk_indices = ttnn.experimental.operations.primary.topk(gate_logits_1SB8, 32)
        ttl_topk_values = ttl_topk_values + top2_mask_11B_64
        mask_B2 = ttnn.eq(expert_mask_11B2, ttl_topk_indices)
        weights = ttnn.sum(ttnn.softmax(ttl_topk_values, dim=-1) * mask_B2, dim=-1)
        print("ttnn took: ", time.time() - start_time)


import time


def test_matmul(device_mesh, use_program_cache):
    reduce_mask = ttnn.from_torch(
        torch.randn(1, 1, 32, 32 * 8),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    reduce_mask = ttnn.to_device(reduce_mask, device_mesh)
    output_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32 * 8, 4096),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    output_11BH = ttnn.to_device(output_11BH, device_mesh)
    for i in range(5):
        start_time = time.time()
        output_reduced = ttnn.matmul(reduce_mask, output_11BH)
        print("ttnn matmul took: ", time.time() - start_time)

        start_time = time.time()
        output_reduced = ttnn.experimental.operations.primary.matmul(reduce_mask, output_11BH)
        print("ttnn experimental matmul took: ", time.time() - start_time)


def test_eq(device_mesh):
    a_torch = torch.full((1, 1, 32, 2), fill_value=1)
    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    b_torch = torch.full((1, 1, 32, 2), fill_value=1)
    b = ttnn.from_torch(
        b_torch,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    print(ttnn.to_torch(ttnn.eq(a, b), mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0])
    # print(torch.eq(a_torch, b_torch))
