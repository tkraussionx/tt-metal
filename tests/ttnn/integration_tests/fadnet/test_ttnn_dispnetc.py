# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_fadnet.reference.dispnetc import DispNetC
from models.experimental.functional_fadnet.tt.ttnn_dispnetc import TtDispNetC

import ttnn
import tt_lib
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, DispNetC):
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            conv1_weight, conv1_bias = model.c1, model.b1
            update_ttnn_module_args(ttnn_module_args.c1)
            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )

            parameters["res1"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                conv3 = block[5]
                bn3 = block[6]

                ttnn_module_args["res1"]["resblock_1_conv1"] = ttnn_module_args["res1"]
                ttnn_module_args["res1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res1"]["resblock_1_conv1"])
                parameters["res1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res1"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res1"]["resblock_2_conv2"] = ttnn_module_args["res1"]
                ttnn_module_args["res1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res1"]["resblock_2_conv2"])
                parameters["res1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res1"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["res1"]["resblock_sc_conv"] = ttnn_module_args["res1"]
                ttnn_module_args["res1"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                update_ttnn_module_args(ttnn_module_args["res1"]["resblock_sc_conv"])
                parameters["res1"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["res1"]["resblock_sc_conv"], return_parallel_config=True
                )

            parameters["res2"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                conv3 = block[5]
                bn3 = block[6]

                ttnn_module_args["res2"]["resblock_1_conv1"] = ttnn_module_args["res2"]
                ttnn_module_args["res2"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res2"]["resblock_1_conv1"])
                parameters["res2"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res2"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res2"]["resblock_2_conv2"] = ttnn_module_args["res2"]
                ttnn_module_args["res2"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res2"]["resblock_2_conv2"])
                parameters["res2"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res2"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["res2"]["resblock_sc_conv"] = ttnn_module_args["res2"]
                ttnn_module_args["res2"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                update_ttnn_module_args(ttnn_module_args["res2"]["resblock_sc_conv"])
                parameters["res2"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["res2"]["resblock_sc_conv"], return_parallel_config=True
                )

            parameters["res3"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res3"]["resblock_1_conv1"] = ttnn_module_args["res3"]
                ttnn_module_args["res3"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res3"]["resblock_1_conv1"])
                parameters["res3"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res3"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res3"]["resblock_2_conv2"] = ttnn_module_args["res3"]
                ttnn_module_args["res3"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res3"]["resblock_2_conv2"])
                parameters["res3"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res3"]["resblock_2_conv2"], return_parallel_config=True
                )

            parameters["res3_1"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res3_1"]["resblock_1_conv1"] = ttnn_module_args["res3_1"]
                ttnn_module_args["res3_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res3_1"]["resblock_1_conv1"])
                parameters["res3_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res3_1"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res3_1"]["resblock_2_conv2"] = ttnn_module_args["res3_1"]
                ttnn_module_args["res3_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res3_1"]["resblock_2_conv2"])
                parameters["res3_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res3_1"]["resblock_2_conv2"], return_parallel_config=True
                )

            parameters["res4"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                conv3 = block[5]
                bn3 = block[6]

                ttnn_module_args["res4"]["resblock_1_conv1"] = ttnn_module_args["res4"]
                ttnn_module_args["res4"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res4"]["resblock_1_conv1"])
                parameters["res4"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res4"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res4"]["resblock_2_conv2"] = ttnn_module_args["res4"]
                ttnn_module_args["res4"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res4"]["resblock_2_conv2"])
                parameters["res4"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res4"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["res4"]["resblock_sc_conv"] = ttnn_module_args["res4"]
                ttnn_module_args["res4"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                update_ttnn_module_args(ttnn_module_args["res4"]["resblock_sc_conv"])
                parameters["res4"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["res4"]["resblock_sc_conv"], return_parallel_config=True
                )

            parameters["res4_1"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res4_1"]["resblock_1_conv1"] = ttnn_module_args["res4_1"]
                ttnn_module_args["res4_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res4_1"]["resblock_1_conv1"])
                parameters["res4_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res4_1"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res4_1"]["resblock_2_conv2"] = ttnn_module_args["res4_1"]
                ttnn_module_args["res4_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res4_1"]["resblock_2_conv2"])
                parameters["res4_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res4_1"]["resblock_2_conv2"], return_parallel_config=True
                )

            parameters["res5"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                conv3 = block[5]
                bn3 = block[6]

                ttnn_module_args["res5"]["resblock_1_conv1"] = ttnn_module_args["res5"]
                ttnn_module_args["res5"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res5"]["resblock_1_conv1"])
                parameters["res5"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res5"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res5"]["resblock_2_conv2"] = ttnn_module_args["res5"]
                ttnn_module_args["res5"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res5"]["resblock_2_conv2"])
                parameters["res5"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res5"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["res5"]["resblock_sc_conv"] = ttnn_module_args["res5"]
                ttnn_module_args["res5"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                update_ttnn_module_args(ttnn_module_args["res5"]["resblock_sc_conv"])
                parameters["res5"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["res5"]["resblock_sc_conv"], return_parallel_config=True
                )

            parameters["res5_1"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res5_1"]["resblock_1_conv1"] = ttnn_module_args["res5_1"]
                ttnn_module_args["res5_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res5_1"]["resblock_1_conv1"])
                parameters["res5_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res5_1"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res5_1"]["resblock_2_conv2"] = ttnn_module_args["res5_1"]
                ttnn_module_args["res5_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res5_1"]["resblock_2_conv2"])
                parameters["res5_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res5_1"]["resblock_2_conv2"], return_parallel_config=True
                )

            parameters["res6"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                conv3 = block[5]
                bn3 = block[6]

                ttnn_module_args["res6"]["resblock_1_conv1"] = ttnn_module_args["res6"]
                ttnn_module_args["res6"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res6"]["resblock_1_conv1"])
                parameters["res6"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res6"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res6"]["resblock_2_conv2"] = ttnn_module_args["res6"]
                ttnn_module_args["res6"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res6"]["resblock_2_conv2"])
                parameters["res6"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res6"]["resblock_2_conv2"], return_parallel_config=True
                )

                ttnn_module_args["res6"]["resblock_sc_conv"] = ttnn_module_args["res6"]
                ttnn_module_args["res6"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
                weight3, bias3 = fold_batch_norm2d_into_conv2d(conv3, bn3)
                update_ttnn_module_args(ttnn_module_args["res6"]["resblock_sc_conv"])
                parameters["res6"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    weight3, bias3, ttnn_module_args["res6"]["resblock_sc_conv"], return_parallel_config=True
                )

            parameters["res6_1"] = {}
            for block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res6_1"]["resblock_1_conv1"] = ttnn_module_args["res6_1"]
                ttnn_module_args["res6_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res6_1"]["resblock_1_conv1"])
                parameters["res6_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res6_1"]["resblock_1_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res6_1"]["resblock_2_conv2"] = ttnn_module_args["res6_1"]
                ttnn_module_args["res6_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res6_1"]["resblock_2_conv2"])
                parameters["res6_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res6_1"]["resblock_2_conv2"], return_parallel_config=True
                )

            ttnn_module_args.pred_flow6["weights_dtype"] = ttnn.bfloat8_b
            pred_flow6_weight = model.pred_flow6
            update_ttnn_module_args(ttnn_module_args.pred_flow6)
            parameters["pred_flow6"], pred_flow6_parallel_config = preprocess_conv2d(
                pred_flow6_weight, None, ttnn_module_args.pred_flow6, return_parallel_config=True
            )

            return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
def test_dispnetc(device, reset_seeds, model_location_generator):
    state_dict = torch.load("tests/ttnn/integration_tests/fadnet/fadnet.pth")
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("module.dispnetc"))}

    torch_model = DispNetC()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 6, 960, 576)  # Batch size of 1, 64 input channels, 160x160 height and width
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtDispNetC(parameters)

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
    output_tensor = output_tensor.reshape(1, 80, 80, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
