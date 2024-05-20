import time
import ttnn
import torch
from ttnn import ReplicateTensorToMesh


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
        print(output_reduced)
        tt = time.time() - start_time

        start_time = time.time()
        output_reduced = ttnn.experimental.operations.primary.matmul(reduce_mask, output_11BH)
        print(output_reduced)
        print("ttnn experimental matmul took: ", tt / (time.time() - start_time))


def test_sharded_matmul(device):
    input_i_1SBH = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    input_i_1SBH = ttnn.to_memory_config(
        input_i_1SBH,
        ttnn.create_sharded_memory_config(
            shape=(32, int(4096 / 32)),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    gates_HE = ttnn.from_torch(
        torch.randn(1, 1, 4096, 64), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    gate_logits_1SBE = ttnn.experimental.operations.primary.matmul(input_i_1SBH, gates_HE)

    print("done", gate_logits_1SBE)
    assert True
