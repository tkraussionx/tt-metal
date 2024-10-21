# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger
import sys
from models.experimental.functional_yolov7.reference.model import Yolov7_model
import models.experimental.functional_yolov7.reference.yolov7_utils
import models.experimental.functional_yolov7.reference.yolov7_model
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from models.experimental.functional_yolov7.ttnn.tt_yolov7 import ttnn_yolov7
from tests.ttnn.utils_for_testing import assert_with_pcc

sys.modules["models.common"] = sys.modules["models.experimental.functional_yolov7.reference.yolov7_utils"]
sys.modules["models.yolo"] = sys.modules["models.experimental.functional_yolov7.reference.yolov7_model"]


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, Yolov7_model):
            print(model.model[0].conv)

            parameters["0"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[0].conv, model.model[0].bn)
            parameters["0"]["weight"] = ttnn.from_torch(
                conv_weight, dtype=ttnn.bfloat16
            )  # bfloat8_b, layout=ttnn.TILE_LAYOUT)
            parameters["0"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16  # bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

            parameters["1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[1].conv, model.model[1].bn)
            parameters["1"]["weight"] = ttnn.from_torch(
                conv_weight, dtype=ttnn.bfloat16
            )  # bfloat8_b, layout=ttnn.TILE_LAYOUT)
            parameters["1"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16  # bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

            parameters["2"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[2].conv, model.model[2].bn)
            parameters["2"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["2"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["3"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[3].conv, model.model[3].bn)
            parameters["3"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["3"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["4"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[4].conv, model.model[4].bn)
            parameters["4"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["4"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["5"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[5].conv, model.model[5].bn)
            parameters["5"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["5"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["6"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[6].conv, model.model[6].bn)
            parameters["6"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["6"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["7"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[7].conv, model.model[7].bn)
            parameters["7"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["7"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["8"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[8].conv, model.model[8].bn)
            parameters["8"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["8"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["9"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[9].conv, model.model[9].bn)
            parameters["9"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["9"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["11"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[11].conv, model.model[11].bn)
            parameters["11"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["11"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["13"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[13].conv, model.model[13].bn)
            parameters["13"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["13"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["14"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[14].conv, model.model[14].bn)
            parameters["14"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["14"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["15"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[15].conv, model.model[15].bn)
            parameters["15"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["15"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


def create_yolov7_model_parameters(model, input_tensor, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)
    # parameters["l1"] = {}
    # parameters["l1"]["weight"] = model.l1.weight
    # parameters["l1"]["bias"] = model.l1.bias
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov7(device, reset_seeds):
    def load_weights(model, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)

    torch_model = Yolov7_model()

    # print("torch_model: ", torch_model)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    # print("1st layer: \n", torch_model.model[0])

    def hook_fn(module, input, output):
        print(f"Input shape to the layer N,C,H,W: {input[0].shape}")
        print(f"Output shape from the layer: {output.shape}")

    # Register the hook to a specific MaxPool2d layer (e.g., the first one at index 2)
    layer = torch_model.model[15]
    print("layer", layer)
    hook = layer.register_forward_hook(hook_fn)

    torch_input_tensor = torch.randn(1, 3, 640, 640)
    weights_path = "tests/ttnn/integration_tests/yolov7/yolov7.pt"
    load_weights(torch_model, weights_path)
    torch_output_tensor = torch_model(torch_input_tensor)
    print("torch output shape: ", torch_output_tensor.shape)

    # Remove the hook after usage
    hook.remove()

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    # parameters = create_yolov7_model_parameters(torch_model.model, torch_input_tensor, device=device)
    print("parameters:", parameters["5"])

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)  # bfloat8_b, layout=ttnn.TILE_LAYOUT)

    ttnn_model = ttnn_yolov7(device, parameters)
    output = ttnn_model(ttnn_input)
    print("ttnn output shape: ", output.shape)

    output = ttnn.to_torch(output)
    output = torch.reshape(output, (1, 80, 80, 256))
    output = torch.permute(output, (0, 3, 1, 2))

    output = output.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output, pcc=0.99)
