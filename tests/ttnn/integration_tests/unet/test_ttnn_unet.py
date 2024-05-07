# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_unet.reference.unet import UNet
from models.experimental.functional_unet.tt.tt_unet import TtUnet


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, UNet):
            parameters["conv"] = {}
            parameters["conv"]["weight"] = model.conv.weight
            parameters["conv"]["bias"] = model.conv.bias
            return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_unet(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("conv.")))}

    torch_model = UNet()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
        # print(keys[i],values[i].shape)

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.load("decoder1_2_output.pt")
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtUnet(device, parameters, new_state_dict)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 480, 640, 1)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)  # pcc = 0.9924751222494625
