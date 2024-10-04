import ttnn
import torch
import pytest
import math
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "act_shape",
    (
        ## mnist shapes
        [1, 1, 28, 28],
    ),
)
def test(reset_seeds, mesh_device, act_shape):
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    x = torch.randn(act_shape, dtype=torch.bfloat16)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    tt_x1 = ttnn.reshape(tt, (tt.shape[0], 1, 1, 784))

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)

    tt_x2 = ttnn.reshape(x, (x.shape[0], 1, 1, 784))

    tt_output = ttnn.to_torch(tt_x1)
    tt_output_device = ttnn.to_torch(tt_x2)

    assert_with_pcc(tt_output, tt_output_device, 1)
