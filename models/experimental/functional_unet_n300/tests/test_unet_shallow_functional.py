# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from loguru import logger

from models.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.experimental.functional_unet_n300.unet_utils import create_unet_models, create_unet_input_tensors

import tt_lib as ttl


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("perf_mode", [True])
@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
def test_unet_pcc(device_mesh, perf_mode, batch, groups):
    with torch.no_grad():
        torch.manual_seed(0)
        num_devices = 1 if isinstance(device_mesh, ttnn.Device) else device_mesh.get_num_devices()

        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

        # Create initial parameters
        torch_input_tensor_tt, ttnn_input_tensor = create_unet_input_tensors(
            device_mesh, batch, groups, inputs_mesh_mapper
        )
        torch_model, ttnn_model = create_unet_models(
            device_mesh,
            batch,
            groups,
            torch_input_tensor_tt,
            weights_mesh_mapper=weights_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

        torch_input_tensor = torch.randn(num_devices * batch, 4 * groups, 1056, 160)
        # Run torch golden result
        torch_output_tensor = torch_model(torch_input_tensor)

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
