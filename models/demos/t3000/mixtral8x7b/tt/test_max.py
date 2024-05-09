import ttnn
import torch
from ttnn import ReplicateTensorToMesh


def test_max(device_mesh):
    gate_logits_1SB8 = ttnn.from_torch(
        torch.randn(1, 1, 32, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    gate_logits_1SB8 = ttnn.to_device(gate_logits_1SB8, device_mesh)
    weights_ex0_1SB1 = ttnn.max(gate_logits_1SB8, dim=3)
    print(weights_ex0_1SB1)


def test_all_gather(device_mesh):
    results_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    results_11BH = ttnn.to_device(results_11BH, device_mesh)
    # results_11BH = [device_tensor for device_tensor in ttnn.get_device_tensors(results_11BH)]
    # hamiltonian_ring_indices = [0, 7, 6, 1, 2, 5, 4, 3]
    # results_11BH = [results_11BH[i] for i in hamiltonian_ring_indices]
    output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
    print("done")


def test_dev(all_devices):
    from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000

    devices = get_devices_for_t3000(all_devices, 8)
    print([d.id() for d in devices])


def test_clone(device_mesh):
    results_11BH = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    results_11BH = ttnn.to_device(results_11BH, device_mesh)
    results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(results_11BH)


def test_create_heads(device_mesh):
    xqkv_fused = ttnn.from_torch(
        torch.randn(1, 1, 32, 768),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    (
        q_heads_14BD,
        k_heads_11BD,
        v_heads_11BD,
    ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=4,
        num_kv_heads=1,
        transpose_k_heads=False,
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(q_heads_14BD, k_heads_11BD, v_heads_11BD)


def test_sharded_matmul(device_mesh):
    q_heads_1B4D = ttnn.from_torch(
        torch.randn(1, 32, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    keys_1BDP = ttnn.from_torch(
        torch.randn(1, 32, 128, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    q_heads_1B4D = ttnn.to_device(q_heads_1B4D, device_mesh)
    keys_1BDP = ttnn.to_device(keys_1BDP, device_mesh)

    q_heads_1B4D = ttnn.to_memory_config(
        q_heads_1B4D,
        ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    print(ttnn.get_memory_config(q_heads_1B4D).is_sharded())
    print(ttnn.is_sharded(q_heads_1B4D))

    keys_1BDP = ttnn.to_memory_config(
        keys_1BDP,
        ttnn.create_sharded_memory_config(
            shape=(128, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # tt::operations::primary::MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(x=8,y=4),in0_block_w=1,out_subblock_h=1,out_subblock_w=1,per_core_M=1,per_core_N=1)
    program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
    )
    attn_1B4P = ttnn.matmul(
        q_heads_1B4D,
        keys_1BDP,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=compute_kernel_attn,
        # program_config = program_config
    )

    print(attn_1B4P)


def test_4b_tensor(device_mesh):
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    tensor = ttnn.to_device(tensor, device_mesh)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    x = ttnn.to_device(x, device_mesh)
    tensor = ttnn.matmul(
        x,
        tensor,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        ),
        use_1d_systolic_array=True,
    )
    print(tensor)


from models.utility_functions import (
    comp_pcc,
)


def test_sliced_softmax(device_mesh):
    tensor = ttnn.from_torch(
        torch.randn(1, 32, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    tensor = ttnn.to_device(tensor, device_mesh)
    tensor = tensor[:, :, :, :1]
    tensor_torch = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    tensor = ttnn.softmax(tensor, dim=-1)
    print(tensor)
    passing, pcc_message = comp_pcc(
        ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]), torch.softmax(tensor_torch, dim=-1), 0.99
    )
    print(passing, pcc_message)
    # assert (all([device_tensor.shape==tensor.shape for device_tensor in ttnn.get_device_tensors(tensor)]))


def test_softmax(device):
    tensor = ttnn.from_torch(
        torch.zeros(1, 1, 32, 1).repeat(1, 32, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tensor_sharded = ttnn.to_memory_config(
        tensor,
        ttnn.create_sharded_memory_config(
            shape=(32, 32),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    # tensor = ttnn.transformer.attention_softmax(tensor, head_size=1, attention_mask = None)
    tensor_sharded = ttnn.transformer.attention_softmax(tensor_sharded, head_size=1, attention_mask=None)

    # assert torch.allclose(ttnn.to_torch(tensor), ttnn.to_torch(tensor_sharded))
    torch_tensors = ttnn.to_torch(tensor_sharded)
    print(torch_tensors)
    # assert torch.allclose(torch_tensors[0], torch_tensors[1])


def test_gates_matmul(device_mesh):
    input_i_1SBH = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    gates_H8 = ttnn.from_torch(
        torch.randn(4096, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    gate_logits_1SB8 = ttnn.linear(
        input_i_1SBH,
        gates_H8,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        #     compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        #     math_fidelity=ttnn.MathFidelity.LoFi,
        #     fp32_dest_acc_en=True,
        #     packer_l1_acc=True,
        # ),
        use_1d_systolic_array=True,
        # core_grid = ttnn.CoreGrid(y=1, x=8),
    )
    assert torch.allclose(
        ttnn.to_torch(ttnn.get_device_tensors(gate_logits_1SB8)[0]).to(torch.bfloat16),
        ttnn.to_torch(ttnn.get_device_tensors(input_i_1SBH)[0]).to(torch.bfloat16)
        @ ttnn.to_torch(ttnn.get_device_tensors(gates_H8)[0]).to(torch.bfloat16),
    )


from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


def test_caching(device_mesh):
    wo_torch = torch.randn(512, 4096)
    wo = ttnn.as_tensor(
        wo_torch,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        cache_file_name="wo_cache_again",
    )
    wo = ttnn.to_device(wo, device_mesh)
    print(wo)

    wo_2 = ttnn.as_tensor(
        wo_torch,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        cache_file_name="wo_cache_again",
    )
    wo_2 = ttnn.to_device(wo_2, device_mesh)
    print(wo_2)

    assert torch.allclose(
        ttnn.to_torch(wo, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0)),
        ttnn.to_torch(wo_2, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0)),
    )


def test_repeat(device_mesh):
    ones_1118 = ttnn.from_torch(
        torch.ones(1, 1, 1, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    ones_1118 = ttnn.to_device(ones_1118, device_mesh)
    weights_ex0_1SB1 = ttnn.from_torch(
        torch.randn(1, 1, 32, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    weights_ex0_1SB1 = ttnn.to_device(weights_ex0_1SB1, device_mesh)
    exp_0_repeated = ttnn.matmul(weights_ex0_1SB1, ones_1118)  # , compute_kernel_config=compute_kernel)
    print(ttnn.to_torch(exp_0_repeated, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0)))


def test_cond(device_mesh):
    expert_mask_torch = []
    for i in range(8):
        torch_tensor = torch.zeros(1, 1, 8, 1)
        torch_tensor[:, :, i, :] = 1
        expert_mask_torch.append(torch_tensor)
    expert_mask_torch = torch.cat(expert_mask_torch, dim=3)
    expert_mask = ttnn.from_torch(
        expert_mask_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
    )
    expert_mask = ttnn.to_device(expert_mask, device_mesh)
    cond0 = ttnn.from_torch(
        torch.randn(1, 1, 32, 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    cond0 = ttnn.to_device(cond0, device_mesh)
    cond0 = ttnn.eq(cond0, cond0)
    cond0 = ttnn.matmul(cond0, expert_mask)

    print(ttnn.to_torch(cond0, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0)))
