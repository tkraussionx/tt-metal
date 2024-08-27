# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
import ttnn
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull


def blazeblock(
    x,
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    skip_proj,
    parameters,
    i,
    conv_config,
    device,
    out_height,
    out_width,
):
    print("ITeration count = ", i, " ", x.shape)
    channel_pad = out_channel - in_channel
    if stride == 2:
        if kernel_size == 3:
            h = ttnn.to_torch(x)
            # h = ttnn.pad(x, ((0, 2), (0, 2)), value=0)
            h = F.pad(h, (0, 2, 0, 2), "constant", 0)
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
        else:
            h = ttnn.to_torch(x)
            h = torch.permute(h, (0, 3, 1, 2))
            h = F.pad(h, (1, 2, 1, 2), "constant", 0)
            h = torch.permute(h, (0, 2, 3, 1))
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
            out_height = h.shape[-2]
            out_width = h.shape[-1]

        max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = max_pool(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    else:
        h = x

    if skip_proj:
        if i == 5:
            print("PyTorchConv")
            x = ttnn.to_torch(x).to(torch.float)
            x = torch.reshape(x, (1, 48, 64, 64))
            skip_conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            skip_conv.weight = parameters[i].skip_proj.weight
            skip_conv.bias = parameters[i].skip_proj.bias
            x = skip_conv(x)
            out_height = x.shape[-2]
            out_width = x.shape[-1]
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            x = ttnn.permute(x, (0, 2, 3, 1))

        else:
            print("TTNN CONV")
            weight = ttnn.from_torch(
                parameters[i].skip_proj.weight, dtype=ttnn.bfloat16
            )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

            bias = ttnn.from_torch(
                parameters[i].skip_proj.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
            )  # , memory_config = ttnn.L1_MEMORY_CONFIG)
            print("skip proj input shape :", x.shape)
            # print(
            #     (
            #         "Iteration :",
            #         i,
            #         x.shape[0],
            #         in_channel,
            #         out_channel,
            #         x.shape[-2],
            #         x.shape[-1],
            #         1,
            #         1,
            #         1,
            #         1,
            #         0,
            #         0,
            #         1,
            #         True,
            #         None,
            #         False,
            #     ),
            #     ",",
            # )
            [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                in_channels=in_channel,
                out_channels=out_channel,
                device=device,
                bias_tensor=bias,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                batch_size=x.shape[0],
                input_height=x.shape[-2],
                input_width=x.shape[-1],
                conv_config=conv_config,
                conv_op_cache={},
                debug=None,
                groups=1,
            )

            print("skip proj output shape :", x.shape)

    elif channel_pad > 0:
        x = ttnn.pad(x, (0, 0, 0, 0), value=0)

    weight = ttnn.from_torch(
        parameters[i].convs[0].weight, dtype=ttnn.bfloat16
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

    bias = ttnn.from_torch(
        parameters[i].convs[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
    )  # , memory_config = ttnn.L1_MEMORY_CONFIG)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        # weights_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,  # 16 if h.shape[1] < 16 else 32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT)
    if i == 5 or i == 9:
        # print("Pytorch conv1")
        print("PyTorchConv")
        h = ttnn.to_torch(h).to(torch.float)

        # h = torch.reshape(h, (1, h.shape[-1], out_height, out_width))
        h = torch.permute(h, (0, 3, 1, 2))
        conv5 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channel,
        )
        conv5.weight = parameters[i].convs[0].weight
        conv5.bias = parameters[i].convs[0].bias
        h = conv5(h)
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        # print("Input shape for conv1 :", h.shape)
        # if i == 16:
        #    print("Out height and width :", out_height," ",out_width)
        print("TTNN CONV")
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            # weight_tensor=ttnn.from_device(parameters[i].convs[0].weight),
            weight_tensor=weight,
            in_channels=in_channel,
            out_channels=in_channel,
            device=device,
            # bias_tensor=ttnn.from_device(parameters[i].convs[0].bias),
            bias_tensor=bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=in_channel,
        )
        print("Output shape for conv1 :", h.shape)

    weight = ttnn.from_torch(
        parameters[i].convs[1].weight, dtype=ttnn.bfloat16
    )  # , device = device, layout = ttnn.TILE_LAYOUT, memory_config = ttnn.L1_MEMORY_CONFIG)
    # weight = ttnn.permute(ttnn.from_device(weight), (2, 3, 0, 1))
    bias = ttnn.from_torch(
        parameters[i].convs[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
    )  # , device = device, layout = ttnn.TILE_LAYOUT, memory_config = ttnn.L1_MEMORY_CONFIG)

    if i == 2 or i == 3 or i == 4 or i == 5 or i == 9:
        # print("Pytorch conv 2")
        h = ttnn.to_torch(h).to(torch.float)
        # if i == 9:
        #    print("Shape of h in 9:", h.shape)
        print("PyTorchConv")
        if i != 5 and i != 9:
            h = torch.reshape(h, (1, h.shape[-1], out_height, out_width))
        conv2 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv2.weight = parameters[i].convs[1].weight
        conv2.bias = parameters[i].convs[1].bias
        h = conv2(h)
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        # print("Input shape for conv2 :", h.shape)
        print("TTNN CONV")
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            # weight_tensor=ttnn.from_device(parameters[i].convs[1].weight),
            weight_tensor=weight,
            in_channels=in_channel,
            out_channels=out_channel,
            device=device,
            # bias_tensor=ttnn.from_device(parameters[i].convs[1].bias),
            bias_tensor=bias,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
        print("Output shape for conv2 :", h.shape)
        h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        h = ttnn.reshape(h, (h.shape[0], out_height, out_width, h.shape[-1]))
        # h = ttnn.to_layout(h, layout = ttnn.TILE_LAYOUT)
        h = ttnn.to_torch(h)
        h = torch.permute(h, (0, 3, 1, 2))
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16)

    # x = ttnn.to_layout(x, dtype = ttnn.bfloat16, layout = ttnn.TILE_LAYOUT)

    x = ttnn.to_torch(x)
    x = ttnn.from_torch(x, device=device)
    if i == 9:
        print("Shape of x before reshape :", x.shape)
    if i == 9:
        out_height = 32
        out_width = 32
        x = ttnn.reshape(ttnn.from_device(x), (1, out_height, out_width, x.shape[-1]))
    else:
        x = ttnn.reshape(ttnn.from_device(x), (1, out_height, out_width, x.shape[-1]))
    x = ttnn.permute(ttnn.to_device(x, device=device), (0, 3, 1, 2))  # n, c, h, w -> n, w, c, h -> 0, 2, 3, 1
    # x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    h = ttnn.to_torch(h)
    x = ttnn.to_torch(x)
    if i == 9:
        print("Shape of x and h", h.shape, " ", x.shape)
    temp = h + x
    temp = ttnn.from_torch(temp, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return ttnn.permute(ttnn.relu(temp), (0, 2, 3, 1)), out_height, out_width
