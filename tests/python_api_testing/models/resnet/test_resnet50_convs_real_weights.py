from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import numpy as np

import tt_lib

from models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
)
from utils import (
    conv3x3,
    conv1x1,
    fold_bn_to_conv,
    fold_bn_to_conv_weights_bias,
)
from tests.python_api_testing.models.resnet.metalResnetBlock50 import compute_conv_output_shape, TtResnetConv, _nearest_32, format_tensor, _nearest_y

import torch
import torch.nn as nn
from torchvision import models

# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    (3136, 64, 64) : {"compute_with_storage_grid_size" : (2,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 1,
                        },

    (3136, 64, 256) : {"compute_with_storage_grid_size" : (4,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 2,
                        },
    (3136, 256, 64) : {"compute_with_storage_grid_size" : (2,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (3136, 256, 128) : {"compute_with_storage_grid_size" : (4,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (800, 128, 512) : {"compute_with_storage_grid_size" : (4,2),
                    "in0_block_w" : 4,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 13,
                    "per_core_N": 4,
                },
    (800, 512, 128) : {"compute_with_storage_grid_size" : (4,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (800, 512, 256) : {"compute_with_storage_grid_size" : (8,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (224, 256, 1024) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 4,
                },
    (224, 1024, 256) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 1,
                },
    (224, 1024, 512) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
    (64, 512, 2048) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 8,
                },
    (64, 2048, 512) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 64,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
}

hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv = {
    (3136, 64) : [128, 64, 128, 64] ,
    (800, 128) : [128, 128, 128, 64] ,
    (224, 256) : [64, 128, 64, 128],
    (64, 512) : [32, 64, 32, 64] ,
}


def make_conv_bn_pairs_in_one_resnet_block(
    inplanes, planes, base_address, state_dict, stride=1
):
    norm_layer = nn.BatchNorm2d
    expansion: int = 4
    base_width = 64.0
    dilation = 1
    groups = 1
    width = int(planes * (base_width / 64.0)) * groups

    conv1_weight = state_dict[f"{base_address}.conv1.weight"]
    conv1_bias = None
    conv1 = conv1x1(
        inplanes, width, state_dict=state_dict, base_address=f"{base_address}.conv1"
    )

    bn1 = norm_layer(width)
    bn1.weight = nn.Parameter(state_dict[f"{base_address}.bn1.weight"])
    bn1.bias = nn.Parameter(state_dict[f"{base_address}.bn1.bias"])
    bn1.running_mean = nn.Parameter(state_dict[f"{base_address}.bn1.running_mean"])
    bn1.running_var = nn.Parameter(state_dict[f"{base_address}.bn1.running_var"])
    bn1.num_batches_tracked = nn.Parameter(
        state_dict[f"{base_address}.bn1.num_batches_tracked"], requires_grad=False
    )
    bn1.eval()

    conv2_weight = state_dict[f"{base_address}.conv2.weight"]
    conv2_bias = None
    conv2 = conv3x3(
        width,
        width,
        stride,
        groups,
        dilation,
        state_dict=state_dict,
        base_address=f"{base_address}.conv2",
    )

    bn2 = norm_layer(width)
    bn2.weight = nn.Parameter(state_dict[f"{base_address}.bn2.weight"])
    bn2.bias = nn.Parameter(state_dict[f"{base_address}.bn2.bias"])
    bn2.running_mean = nn.Parameter(state_dict[f"{base_address}.bn2.running_mean"])
    bn2.running_var = nn.Parameter(state_dict[f"{base_address}.bn2.running_var"])
    bn2.num_batches_tracked = nn.Parameter(
        state_dict[f"{base_address}.bn2.num_batches_tracked"], requires_grad=False
    )
    bn2.eval()

    conv3_weight = state_dict[f"{base_address}.conv3.weight"]
    conv3_bias = None
    conv3 = conv1x1(
        width,
        planes * expansion,
        state_dict=state_dict,
        base_address=f"{base_address}.conv3",
    )

    bn3 = norm_layer(planes * expansion)
    bn3.weight = nn.Parameter(state_dict[f"{base_address}.bn3.weight"])
    bn3.bias = nn.Parameter(state_dict[f"{base_address}.bn3.bias"])
    bn3.running_mean = nn.Parameter(state_dict[f"{base_address}.bn3.running_mean"])
    bn3.running_var = nn.Parameter(state_dict[f"{base_address}.bn3.running_var"])
    bn3.num_batches_tracked = nn.Parameter(
        state_dict[f"{base_address}.bn3.num_batches_tracked"], requires_grad=False
    )
    bn3.eval()

    return [(conv1, bn1), (conv2, bn2), (conv3, bn3)]


def test_resnet50_convs_with_folded_batch_norm(device):
    with torch.no_grad():
        torch.manual_seed(1234)
        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()
        state_dict = torch_resnet50.state_dict()
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        layer_planes = [64, 128, 256, 512]
        layer_blocks = [3, 4, 6, 3]
        layer_strides = [1, 2, 2, 2]
        conv_bn_pairs = []
        inplanes = 64
        base_address_with_dot = ""
        expansion = 4
        norm_layer = nn.BatchNorm2d

        # first conv and batch norm
        conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight = nn.Parameter(state_dict[f"{base_address_with_dot}conv1.weight"])
        bn1 = norm_layer(inplanes)  # batch norm
        bn1.weight = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.weight"])
        bn1.bias = nn.Parameter(state_dict[f"{base_address_with_dot}bn1.bias"])
        bn1.running_mean = nn.Parameter(
            state_dict[f"{base_address_with_dot}bn1.running_mean"]
        )
        bn1.running_var = nn.Parameter(
            state_dict[f"{base_address_with_dot}bn1.running_var"]
        )
        bn1.num_batches_tracked = nn.Parameter(
            state_dict[f"{base_address_with_dot}bn1.num_batches_tracked"],
            requires_grad=False,
        )
        bn1.eval()
        conv_bn_pairs.append((conv1, bn1))
        maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )  # will run maxpool to get the correct shape for next conv
        for i, name in enumerate(layer_names):
            planes = layer_planes[i]
            stride = layer_strides[i]
            blocks = layer_blocks[i]
            conv_bn_pairs.extend(
                make_conv_bn_pairs_in_one_resnet_block(
                    inplanes,
                    planes,
                    f"{base_address_with_dot}{name}.0",
                    state_dict,
                    stride,
                )
            )
            inplanes = planes * expansion
            for _ in range(1, blocks):
                conv_bn_pairs.extend(
                    make_conv_bn_pairs_in_one_resnet_block(
                        inplanes,
                        planes,
                        f"{base_address_with_dot}{name}.{_}",
                        state_dict,
                        1,
                    )
                )

        x_shape = [1, 3, 224, 224]
        memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
        for i, conv_bn_pair in enumerate(conv_bn_pairs):
            conv = conv_bn_pair[0]
            bn = conv_bn_pair[1]
            x = torch.randn(x_shape, dtype=torch.bfloat16).float()
            x_nhwc = torch.permute(x, (0, 2, 3, 1))
            conv_input_shape_nhwc = x_nhwc.shape
            # Run pytorch golden reference -> Conv followed by BN
            x_golden = conv(x)
            x_golden = bn(x_golden)

            # Fold batchnorm into conv weights and bias
            conv_weight, conv_bias = fold_bn_to_conv_weights_bias(conv.weight, bn)

            # # Run pytorch conv with folded bn
            # conv.weight = nn.Parameter(conv_weight)
            # conv.bias = nn.Parameter(conv_bias)
            # x_pytorch_folded_bn = conv(x)

            # # Compare pytorch golden vs pytorch with folded bn
            # assert x_pytorch_folded_bn.shape == x_golden.shape
            # passing_pcc, output_pcc = comp_pcc(x_golden, x_pytorch_folded_bn, 0.99)
            # print(
            #     "Passing (Pytorch golden vs Pytorch conv with folden batchnorm)=",
            #     passing_pcc,
            # )
            # print("Output pcc=", output_pcc)
            # assert passing_pcc

            # Run conv on device with folded batch norm
            conv_params = [
                conv.out_channels,
                conv.in_channels,
                conv.kernel_size[0],
                conv.kernel_size[1],
                conv.stride[0],
                conv.stride[1],
                conv.padding[0],
                conv.padding[1],
                1,
                1,
            ]
            K, C, R, S, stride_h, stride_w, pad_h, pad_w, dilation, groups = [conv_params[i] for i in range(10)]
            print("K=", K, "C=", C, "H=", conv_input_shape_nhwc[1], "W=", conv_input_shape_nhwc[2], "R=", R, "S=", S, "U=", stride_h, "V=", stride_w, "P_H=", pad_h, "P_W=", pad_w)

            is_1x1_conv = R == 1 and S == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0
            is_first_conv = R == 7 and S == 7
            print("is_first_conv", is_first_conv)
            conv_params = [K, C, R, S, stride_h, stride_w, pad_h, pad_w, 1, 1]
            conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
            print("Conv output shape - ", conv_output_shape)
            conv_as_mm_padded_act_height = _nearest_32(conv_output_shape[1] * conv_output_shape[2])

            if (is_1x1_conv):
                matmul_config = None
                assert (conv_as_mm_padded_act_height, C, K) in hardcoded_matmul_config_conv
                print("Setting matmul config for 1x1 conv")
                matmul_config = hardcoded_matmul_config_conv[(conv_as_mm_padded_act_height, C, K)]
                # 1x1 conv with stride 1 padding 0 is run using regular matmul
                conv = TtResnetConv(conv_weight.reshape(-1).tolist(), conv_params, device, [1, 1], [1, 1], [1, 1], conv_bias.tolist())
            elif is_first_conv:
                conv = TtResnetConv(conv_weight.reshape(-1).tolist(),
                                      conv_params,
                                      device,
                                      [128, 128],
                                      [128, 64],
                                      [128, 64],
                                      conv_bias.tolist(),
                                      8,
                                      True)
            else:
                assert (conv_as_mm_padded_act_height, K) in hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv
                [act_block_h_datums, weight_block_w_datums, out_subblock_h_datums, out_subblock_w_datums] = hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv[(conv_as_mm_padded_act_height, K)]
                conv = TtResnetConv(conv_weight.reshape(-1).tolist(),
                                    conv_params,
                                    device,
                                    [act_block_h_datums, C*S], [C*S, weight_block_w_datums],
                                    [out_subblock_h_datums, out_subblock_w_datums],
                                    conv_bias.tolist())

            x_nhwc = tt_lib.tensor.Tensor(
                                    x_nhwc.reshape(-1).tolist(),
                                    x_nhwc.shape,
                                    tt_lib.tensor.DataType.BFLOAT16,
                                    tt_lib.tensor.Layout.ROW_MAJOR)
            if is_first_conv:
                #Pre-pad first conv
                act_shape_height_width_channel_padded = [conv_input_shape_nhwc[0], conv_input_shape_nhwc[1] + 6, conv_input_shape_nhwc[2] + 7, _nearest_y(conv_input_shape_nhwc[3], 16)]
                x_nhwc = x_nhwc.pad(act_shape_height_width_channel_padded, (0, 3, 3, 0), 0)

            conv_input_on_device = x_nhwc.to(device, memory_config)

            if (is_1x1_conv):
                # convert activation RM to tile layout
                conv_input_on_device = conv_input_on_device.reshape(1, 1, conv_input_shape_nhwc[1]*conv_input_shape_nhwc[2], conv_input_shape_nhwc[3])
                conv_input_on_device = format_tensor(conv_input_on_device, tt_lib.tensor.Layout.TILE, device, memory_config)

            output_on_device = conv(conv_input_on_device, False)


            # convert matmul tiled output to RM
            assert(output_on_device.layout() == tt_lib.tensor.Layout.TILE)
            output_on_device = format_tensor(output_on_device, tt_lib.tensor.Layout.ROW_MAJOR, device, memory_config)
            output_on_device = output_on_device.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])

            # Copy to host and Compare against pytorch
            out = output_on_device.cpu()
            assert out.layout() == tt_lib.tensor.Layout.ROW_MAJOR

            out_result = out.to_torch()
            # NHWC to NCHW
            out_result = torch.transpose(out_result, 2, 3)
            out_result = torch.transpose(out_result, 1, 2)

            # Compare pytorch golden vs conv with folded batchnorm on device
            assert out_result.shape == x_golden.shape
            passing_pcc, output_pcc = comp_pcc(x_golden, out_result, 0.99)
            print(
                "Passing (Pytorch golden vs Conv with folden batchnorm on device)=",
                passing_pcc,
            )
            print("Output pcc=", output_pcc)
            assert passing_pcc

            if i == 0:
                # run maxpool to get the correct shape for next conv
                x_golden = maxpool(x_golden)
            x_shape = x_golden.shape  # for next iteration
