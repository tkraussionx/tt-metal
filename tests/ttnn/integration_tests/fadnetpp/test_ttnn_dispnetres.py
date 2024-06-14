# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn, Tensor
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_fadnetpp.reference.dispnetres import DispNetRes
from models.experimental.functional_fadnetpp.tt.tt_dispnetres import TtDispNetRes
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def create_custom_preprocessor(device, resblock=True):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, DispNetRes):
            ttnn_module_args["conv1"] = ttnn_module_args.conv1["0"]
            conv1_weight, conv1_bias = torch_model.conv1[0].weight, torch_model.conv1[0].bias
            update_ttnn_module_args(ttnn_module_args["conv1"])
            ttnn_module_args["conv1"]["use_1d_systolic_array"] = True
            ttnn_module_args["conv1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["conv1"]["use_shallow_conv_variant"] = False
            parameters["conv1"], conv1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["conv1"], return_parallel_config=True
            )
            if resblock:
                parameters["conv2"] = {}

                ttnn_module_args["conv2"]["resblock_1_conv1"] = ttnn_module_args["conv2"]["resblock_1_conv1"]
                conv2_weight1, conv2_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.resblock_1_conv1, torch_model.conv2.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_1_conv1"])
                parameters["conv2"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv2_weight1,
                    conv2_bias1,
                    ttnn_module_args["conv2"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv2"]["resblock_2_conv2"] = ttnn_module_args["conv2"]["resblock_2_conv2"]
                conv2_weight2, conv2_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.resblock_2_conv2, torch_model.conv2.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_2_conv2"])
                parameters["conv2"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv2_weight2,
                    conv2_bias2,
                    ttnn_module_args["conv2"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv2"]["resblock_sc_conv"] = ttnn_module_args["conv2"]["shortcut_c"]
                conv2_weight3, conv2_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.shortcut_c, torch_model.conv2.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_sc_conv"])
                parameters["conv2"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv2_weight3,
                    conv2_bias3,
                    ttnn_module_args["conv2"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv3"] = {}

                ttnn_module_args["conv3"]["resblock_1_conv1"] = ttnn_module_args["conv3"]["resblock_1_conv1"]
                conv3_weight1, conv3_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.resblock_1_conv1, torch_model.conv3.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_1_conv1"])
                parameters["conv3"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv3_weight1,
                    conv3_bias1,
                    ttnn_module_args["conv3"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3"]["resblock_2_conv2"] = ttnn_module_args["conv3"]["resblock_2_conv2"]
                conv3_weight2, conv3_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.resblock_2_conv2, torch_model.conv3.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_2_conv2"])
                parameters["conv3"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv3_weight2,
                    conv3_bias2,
                    ttnn_module_args["conv3"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3"]["resblock_sc_conv"] = ttnn_module_args["conv3"]["shortcut_c"]
                conv3_weight3, conv3_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.shortcut_c, torch_model.conv3.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_sc_conv"])
                parameters["conv3"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv3_weight3,
                    conv3_bias3,
                    ttnn_module_args["conv3"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv3_1"] = {}

                ttnn_module_args["conv3_1"]["resblock_1_conv1"] = ttnn_module_args["conv3_1"]["resblock_1_conv1"]
                conv3_1_weight1, conv3_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3_1.resblock_1_conv1, torch_model.conv3_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv3_1"]["resblock_1_conv1"])
                parameters["conv3_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv3_1_weight1,
                    conv3_1_bias1,
                    ttnn_module_args["conv3_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3_1"]["resblock_2_conv2"] = ttnn_module_args["conv3_1"]["resblock_2_conv2"]
                conv3_1_weight2, conv3_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3_1.resblock_2_conv2, torch_model.conv3_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv3_1"]["resblock_2_conv2"])
                parameters["conv3_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv3_1_weight2,
                    conv3_1_bias2,
                    ttnn_module_args["conv3_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                parameters["conv4"] = {}

                ttnn_module_args["conv4"]["resblock_1_conv1"] = ttnn_module_args["conv4"]["resblock_1_conv1"]
                conv4_weight1, conv4_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.resblock_1_conv1, torch_model.conv4.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_1_conv1"])
                parameters["conv4"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv4_weight1,
                    conv4_bias1,
                    ttnn_module_args["conv4"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4"]["resblock_2_conv2"] = ttnn_module_args["conv4"]["resblock_2_conv2"]
                conv4_weight2, conv4_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.resblock_2_conv2, torch_model.conv4.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_2_conv2"])
                parameters["conv4"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv4_weight2,
                    conv4_bias2,
                    ttnn_module_args["conv4"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4"]["resblock_sc_conv"] = ttnn_module_args["conv4"]["shortcut_c"]
                conv4_weight3, conv4_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.shortcut_c, torch_model.conv4.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_sc_conv"])
                parameters["conv4"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv4_weight3,
                    conv4_bias3,
                    ttnn_module_args["conv4"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv4_1"] = {}

                ttnn_module_args["conv4_1"]["resblock_1_conv1"] = ttnn_module_args["conv4_1"]["resblock_1_conv1"]
                conv4_1_weight1, conv4_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4_1.resblock_1_conv1, torch_model.conv4_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_1_conv1"])
                parameters["conv4_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv4_1_weight1,
                    conv4_1_bias1,
                    ttnn_module_args["conv4_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4_1"]["resblock_2_conv2"] = ttnn_module_args["conv4_1"]["resblock_2_conv2"]
                conv4_1_weight2, conv4_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4_1.resblock_2_conv2, torch_model.conv4_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_2_conv2"])
                parameters["conv4_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv4_1_weight2,
                    conv4_1_bias2,
                    ttnn_module_args["conv4_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                parameters["conv5"] = {}

                ttnn_module_args["conv5"]["resblock_1_conv1"] = ttnn_module_args["conv5"]["resblock_1_conv1"]
                conv5_weight1, conv5_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.resblock_1_conv1, torch_model.conv5.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_1_conv1"])
                ttnn_module_args["conv5"]["resblock_1_conv1"]["use_1d_systolic_array"] = True
                ttnn_module_args["conv5"]["resblock_1_conv1"][
                    "conv_blocking_and_parallelization_config_override"
                ] = None
                ttnn_module_args["conv5"]["resblock_1_conv1"]["use_shallow_conv_variant"] = False
                parameters["conv5"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv5_weight1,
                    conv5_bias1,
                    ttnn_module_args["conv5"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv5"]["resblock_2_conv2"] = ttnn_module_args["conv5"]["resblock_2_conv2"]
                conv5_weight2, conv5_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.resblock_2_conv2, torch_model.conv5.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_2_conv2"])
                parameters["conv5"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv5_weight2,
                    conv5_bias2,
                    ttnn_module_args["conv5"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv5"]["resblock_sc_conv"] = ttnn_module_args["conv5"]["shortcut_c"]
                conv5_weight3, conv5_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.shortcut_c, torch_model.conv5.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_sc_conv"])
                parameters["conv5"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv5_weight3,
                    conv5_bias3,
                    ttnn_module_args["conv5"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dispnetc(device, reset_seeds):
    in_planes = 3 * 3 + 1 + 1
    torch_model = DispNetRes(in_planes)
    print("torch_model:", torch_model)
    # for layer in torch_model.children():
    #     print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    state_dict = torch_model
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 11, 960, 576)
    pr0 = torch.randn(1, 1, 960, 576)
    pr1 = torch.randn(1, 1, 480, 288)
    pr2 = torch.randn(1, 1, 240, 144)
    pr3 = torch.randn(1, 1, 120, 72)
    pr4 = torch.randn(1, 1, 60, 36)
    pr5 = torch.randn(1, 1, 30, 18)
    pr6 = torch.randn(1, 1, 15, 9)
    dispnetc_flows = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)
    torch_output_tensor = torch_model(torch_input_tensor, dispnetc_flows)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor, dispnetc_flows),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtDispNetRes(parameters, device, in_planes)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    ttnn_pr0 = ttnn.from_torch(pr0, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr1 = ttnn.from_torch(pr1, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr2 = ttnn.from_torch(pr2, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr3 = ttnn.from_torch(pr3, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr4 = ttnn.from_torch(pr4, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr5 = ttnn.from_torch(pr5, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_pr6 = ttnn.from_torch(pr6, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_dispnetc_flows = (ttnn_pr0, ttnn_pr1, ttnn_pr2, ttnn_pr3, ttnn_pr4, ttnn_pr5, ttnn_pr6)

    output_tensor = ttnn_model(device, input_tensor, ttnn_dispnetc_flows)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
