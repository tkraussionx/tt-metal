# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.demos.falcon40b_prefill.reference.hf_modeling_falcon import FalconForCausalLM, FalconConfig
from models.demos.falcon40b_prefill.tt.falcon_mlp import TtFalconMLP
from models.demos.falcon40b_prefill.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull
from models.demos.falcon40b_prefill.tt.model_utils import memcfg_1d_width_sharded_from_tensor_shape


class PytorchFalconMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.transformer.h[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_FalconMLP_inference(
    devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    pcc,
    model_config,
    tt_cache_path,
    emulate_per_device_fracture,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    config = FalconConfig.from_pretrained(model_name, num_hidden_layers=1)
    hugging_face_reference_model = FalconForCausalLM(config)
    first_layer_weights_path = tt_cache_path / "transformer.h.0.pt"
    state_dict = torch.load(first_layer_weights_path)

    hugging_face_reference_model.transformer.h[0].load_state_dict(torch.load(first_layer_weights_path), strict=False)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    if llm_mode == "decode":
        input_shape = [seq_len, 1, batch, config.hidden_size]
    else:
        input_shape = [batch, 1, seq_len, config.hidden_size]
    mlp_input = (torch.rand(input_shape) * 2) - 1
    layer_num = 0
    base_url = "transformer.h"

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconMLP_model = PytorchFalconMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_FalconMLP_model(mlp_input)

    # TT hardware execution -------------------------------------------------------------
    tt_FalconMLP_model = TtFalconMLP(
        devices,
        state_dict,
        base_url,
        layer_num,
        config.hidden_size,
        model_config,
        tt_cache_path,
        emulate_per_device_fracture,
    )

    tt_mlp_input_host = ttnn.from_torch(mlp_input, layout=ttnn.TILE_LAYOUT, dtype=model_config["LN_MLP_OUTPUT_DTYPE"])
    tt_mlp_input = []
    for device in devices:
        # mem_cfg = ttnn.L1_MEMORY_CONFIG
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        tt_mlp_input.append(ttnn.to_device(tt_mlp_input_host, device=device, memory_config=mem_cfg))

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    if len(tt_out) > 1:
        tt_out = torch.concat([ttnn.to_torch(tt_o) for tt_o in tt_out], -1)
    else:
        tt_out = ttnn.to_torch(tt_out[0])  # all in one matmuls

    # check outputs ----------------------------------------------------------------------
    if emulate_per_device_fracture:
        logger.info("Skipping PCC evaluation for per device fracture emulation")
    else:
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"PCC value: {output_pcc}")

        if does_pass:
            logger.info("Falcon MLP output Passed!")
        else:
            logger.warning("Falcon MLP output Failed!")
            assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "model_version, llm_mode, batch, seq_len, emulate_per_device_fracture",
    [
        # (
        #     "tiiuae/falcon-40b-instruct",
        #     "decode",
        #     32,
        #     1,
        #     False,
        # ),
        ("tiiuae/falcon-40b-instruct", "prefill", 1, 2048, False),
    ],
)
@pytest.mark.parametrize(
    "model_config_str, pcc",
    [
        # ("BFLOAT8_B-SHARDED", 0.85),
        ("BFLOAT16-SHARDED", 0.87)
    ],
)
def test_FalconMLP_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    pcie_devices,
    use_program_cache,
    emulate_per_device_fracture,
):
    model_config = get_model_config(model_config_str)
    compute_grid_size = pcie_devices[0].compute_with_storage_grid_size()
    if emulate_per_device_fracture:
        print(f"Emulating one fracture on 1 device")
        pcie_devices = [pcie_devices[0]]
    elif len(pcie_devices) == 1:
        print(f"Emulating sequentially on 1 device")
        pcie_devices = pcie_devices * 4
    elif len(pcie_devices) < model_config["NUM_DEVICES"]:
        pytest.skip(f"Requires at least {model_config['NUM_DEVICES']} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    run_test_FalconMLP_inference(
        pcie_devices[: model_config["NUM_DEVICES"]],
        model_version,
        llm_mode,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        emulate_per_device_fracture,
        model_location_generator,
    )
