import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "test_input",
    [
        1,
    ],
)
def test_in_place_add(
    test_input,
    t3k_device_mesh,
):
    input_tensor_a = torch.ones((1, 1, 128, 2048))
    input_tensor_b = torch.ones((1, 1, 128, 2048))

    tt_a = ttnn.from_torch(
        input_tensor_a,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=3),
    )
    tt_a = ttnn.to_device(tt_a, t3k_device_mesh)

    ttnn_input_a_tensor = ttnn.from_torch(
        input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=3),
    )
    ttnn_input_b_tensor = ttnn.from_torch(
        input_tensor_b,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_device_mesh, dim=3),
    )
    output = ttnn.add(
        ttnn_input_a_tensor,
        ttnn_input_b_tensor,
        output_tensor=ttnn_input_a_tensor,
    )

    output = ttnn.to_torch(
        output, device=t3k_device_mesh, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=3)
    )

    output1 = ttnn.to_torch(
        ttnn_input_a_tensor, device=t3k_device_mesh, mesh_composer=ttnn.ConcatMeshToTensor(t3k_device_mesh, dim=3)
    )

    print(output)
    print(output1)
    print(output == output1)
    print(f"Output shape: {output.shape}")
