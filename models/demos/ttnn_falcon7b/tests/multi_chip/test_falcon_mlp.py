# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor, strip_state_dict_prefix
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
import transformers

from loguru import logger

PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


def get_model_prefix(layer_index: int = 0):
    return f"transformer.h.{layer_index}.mlp"


@pytest.fixture(scope="module")
def torch_model():
    hugging_face_reference_model = transformers.FalconForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME, low_cpu_mem_usage=True
    ).eval()
    state_dict = hugging_face_reference_model.state_dict()
    mlp_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())

    configuration = transformers.FalconConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    torch_model = transformers.models.falcon.modeling_falcon.FalconMLP(configuration).eval()
    torch_model.load_state_dict(mlp_state_dict)
    return torch_model


@pytest.mark.parametrize(
    "model_name, batch, seq_len, expected_pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            0.99,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@pytest.mark.parametrize(
    "enable_async",
    [True, False],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16384}], indirect=True)
def test_falcon_mlp(
    pcie_device_mesh,
    model_name,
    batch,
    seq_len,
    expected_pcc,
    model_config_str,
    torch_model,
    enable_async,
):
    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(enable_async)
        pcie_device_mesh.get_device(device).enable_program_cache()

    torch.manual_seed(0)

    configuration = transformers.FalconConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    torch_input = (
        torch.rand(batch * pcie_device_mesh.get_num_devices(), 1, seq_len, configuration.hidden_size) * 2
    ) - 1
    torch_output = torch_model(torch_input)

    model_config = get_model_config(model_config_str)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=pcie_device_mesh,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_name}"),
            device=pcie_device_mesh,
            base_file_name=get_model_prefix(),
            weights_mesh_mapper=ReplicateTensorToMesh(pcie_device_mesh),
        ),
    )

    ttnn_model = TtFalconMLP(model_config, parameters)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=model_config["DEFAULT_DTYPE"],
        device=pcie_device_mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
    )
    logger.info("Compiling Model")
    ttnn_model(ttnn_input)
    logger.info("Capturing Trace")
    trace_id = ttnn.begin_trace_capture(pcie_device_mesh, cq_id=0)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn.end_trace_capture(pcie_device_mesh, trace_id, cq_id=0)

    logger.info("Executing Traced Model")
    for i in range(10):
        torch_input = (
            torch.rand(batch * pcie_device_mesh.get_num_devices(), 1, seq_len, configuration.hidden_size) * 2
        ) - 1
        torch_output = torch_model(torch_input)

        ttnn_input_updated = ttnn.from_torch(
            torch_input,
            dtype=model_config["DEFAULT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ShardTensorToMesh(pcie_device_mesh, dim=0),
        )
        ttnn.copy_host_to_device_tensor(ttnn_input_updated, ttnn_input)
        ttnn.execute_trace(pcie_device_mesh, trace_id, cq_id=0, blocking=False)
        passed, pcc = assert_with_pcc(
            torch_output,
            ttnn.to_torch(
                ttnn_output, mesh_composer=ConcatMeshToTensor(pcie_device_mesh, dim=0), device=pcie_device_mesh
            ).to(torch_output.dtype),
            expected_pcc,
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    for device in pcie_device_mesh.get_device_ids():
        pcie_device_mesh.get_device(device).enable_async(False)
