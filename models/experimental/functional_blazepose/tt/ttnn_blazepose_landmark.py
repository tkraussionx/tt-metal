# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
import ttnn
from math import sqrt


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
    itr=-1,
    landmark=False,
):
    channel_pad = out_channel - in_channel
    print("input to blazepose block", x.shape)
    if stride == 2:
        if kernel_size == 3:
            print("inside 1")
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            h = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2)))
            h = F.pad(h, (0, 2, 0, 2), "constant", 0)
            h = torch.permute(h, (0, 2, 3, 1))
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
            out_height = h.shape[-2]
            out_width = h.shape[-1]
        else:
            print("inside 2")
            h = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2)))
            h = F.pad(h, (1, 2, 1, 2), "constant", 0)
            h = torch.permute(h, (0, 2, 3, 1))
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
            out_height = h.shape[-2]
            out_width = h.shape[-1]

        # if True:
        if (not landmark and (i == 5 or i == 16)) or (landmark and (itr == 2 and i == 0) or not (itr == 9 and i != 0)):
            print("TORCH MAXPOOL")
            print("itr", itr)
            print("iteration", i)
            max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            x = ttnn.to_torch(x)
            x = torch.permute(x, (0, 3, 1, 2))
            # x = torch.reshape(x, (x.shape[0], x.shape[-3],x.shape[-2] , x.shape[-1]))
            print("input maxpool", x.shape)
            x = max_pool(x)
            print("output maxpool", x.shape)
            print("=============")
            # x = torch.reshape(x, (x.shape[0], x.shape[-1] ,x.shape[-2], x.shape[3]))
            x = torch.permute(x, (0, 2, 3, 1))
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
        else:
            print("TTNN MAXPOOL")
            print("itr", itr)
            print("iteration", i)
            print("input shape", x.shape)
            x_shape = x.shape
            x = ttnn.to_torch(x)
            x = torch.reshape(x, (x.shape[0], 1, x.shape[-3] * x.shape[-2], x.shape[-1]))
            x = ttnn.from_torch(x, device=device)
            maxpool = ttnn.MaxPool2d(
                kernel_size=(stride, stride),
                stride=(stride, stride),
                padding=(0, 0),
                dilation=(1, 1),
                dtype=ttnn.bfloat16,
                device=device,
                batch_size=x_shape[0],
                input_height=x_shape[-2],
                input_width=x_shape[-2],
                reader_patterns_cache={},
            )
            x = maxpool.copy_input_to_device(x)
            x = maxpool(x)
            print("output shape", x.shape)
            print("=============")
            x = ttnn.to_torch(x)
            x = torch.reshape(x, (x_shape[0], x_shape[1] // 2, x_shape[1] // 2, x_shape[3]))
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    else:
        h = x

    out_height_x = out_height
    out_width_x = out_width
    if skip_proj:
        if i == 5:
            x = ttnn.reshape(x, (x.shape[0], x.shape[-1], x.shape[1], x.shape[2]))
            x = ttnn.to_torch(x).to(torch.float)
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
            out_height_x = x.shape[-2]
            out_width_x = x.shape[-1]
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2] * x.shape[3]))
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            x = ttnn.permute(x, (0, 1, 3, 2))

        else:
            weight = ttnn.from_torch(parameters[i].skip_proj.weight, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(
                parameters[i].skip_proj.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16
            )
            [x, out_height_x, out_width_x, weights_device, bias_device] = ttnn.conv2d(
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
                input_height=x.shape[1],
                input_width=x.shape[2],
                conv_config=conv_config,
                conv_op_cache={},
                debug=False,
                groups=1,
            )

    elif channel_pad > 0:
        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.pad(x, (0, 0, 0, 0, 0, channel_pad), value=0)
        x = torch.permute(x, (0, 2, 3, 1))
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=False,
    )

    weight = ttnn.from_torch(parameters[i].convs[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters[i].convs[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV 0")
    print("itr", itr)
    print("iteration", i)
    print("input to ttnn conv 0", h.shape, h.shape[1], h.shape[2])
    [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=weight,
        in_channels=in_channel,
        out_channels=in_channel,
        device=device,
        bias_tensor=bias,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=1,
        input_height=h.shape[1],
        input_width=h.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=in_channel,
    )
    print("output shape", h.shape, out_height, out_width)
    print("=============")

    if (not landmark and (i == 2 or i == 3 or i == 4 or i == 5)) or (
        landmark
        and (itr == 1 and (i == 2 or i == 3))
        or (itr == 2 and (i == 0 or i == 1 or i == 2 or i == 3))
        or (itr == 3 and i == 0)
        or (itr == 6 and i == 0)
    ):
        h = ttnn.to_torch(h).to(torch.float)
        h = torch.permute(h, (0, 3, 1, 2))
        h = torch.reshape(h, (1, h.shape[1], out_height, out_width))
        conv2 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv2.weight = parameters[i].convs[1].weight
        conv2.bias = parameters[i].convs[1].bias
        print("TORCH CONV 1")
        print("itr", itr)
        print("Iteration", i)
        print("input shape of h", h.shape)
        h = conv2(h)
        out_height = h.shape[2]
        out_width = h.shape[3]
        h = torch.reshape(h, (1, 1, h.shape[1], out_height * out_width))
        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.permute(h, (0, 1, 3, 2))
        print("output of conv", h.shape)
        print("=============")

    else:
        weight = ttnn.from_torch(parameters[i].convs[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters[i].convs[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("TTNN CONV 1")
        print("itr", itr)
        print("Iteration", i)
        print("input shape", h.shape, out_height, out_width)
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            weight_tensor=weight,
            in_channels=in_channel,
            out_channels=out_channel,
            device=device,
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
        print("output shape", h.shape)
        print("=============")

    h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    h = ttnn.reshape(h, (h.shape[0], out_height, out_width, h.shape[-1]))
    h = ttnn.to_torch(h)
    h = torch.permute(h, (0, 3, 1, 2))
    h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device)
    h = ttnn.to_layout(h, layout=ttnn.TILE_LAYOUT)

    x = ttnn.to_torch(x)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    x = ttnn.reshape(ttnn.from_device(x), (x.shape[0], out_height, out_width, x.shape[-1]))
    x = ttnn.permute(ttnn.to_device(x, device=device), (0, 3, 1, 2))  # n, c, h, w -> n, w, c, h -> 0, 2, 3, 1
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

    temp = ttnn.add(h, x)
    out = ttnn.permute(ttnn.relu(temp), (0, 2, 3, 1))
    return ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT), out_height, out_width


#########


def ttnn_basepose_land_mark(
    x,
    parameters,
    device,
):
    torch_conv = True
    x = ttnn.to_torch(x)
    batch = x.shape[0]
    if batch == 0:
        return (
            torch.zeros((0,)),
            torch.zeros((0, 31, 4)),
            torch.zeros((0, 128, 128)),
        )
    # x = F.pad(x, (0, 1, 1, 0), "constant", 0)
    x = F.pad(x, (0, 0, 0, 1, 0, 1, 0, 0), "constant", 0)

    x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    use_shallow_conv_variant = False
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,  # if C < H*W*N h_s else b_s
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=True,
        # packer_l1_accum_enabled=False,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        # transpose_shards=True, #if b_s is True
        reshard_if_not_optimal=False,
        deallocate_activation=False,
        reallocate_halo_output=True,  # migght resolve OOM in
        act_block_h_override=32,
    )

    weight = ttnn.from_torch(parameters.backbone1[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.backbone1[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    x = ttnn.from_device(x)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    print("TTNN CONV")
    print("input shape first conv", x.shape)
    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        in_channels=3,
        out_channels=24,
        device=device,
        bias_tensor=bias,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(0, 0),
        batch_size=1,
        input_height=257,
        input_width=257,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=False,
        groups=1,
    )
    x = ttnn.relu(x)

    hx = ttnn.to_torch(x)
    hx = torch.reshape(hx, (x.shape[0], out_height, out_width, x.shape[-1]))
    hx = ttnn.from_torch(hx, dtype=ttnn.bfloat16)
    x = ttnn.to_layout(hx, layout=ttnn.ROW_MAJOR_LAYOUT)

    for i in range(2, 4):
        x, out_height, out_width = blazeblock(
            x,
            24,
            24,
            3,
            1,
            1,
            False,
            parameters.backbone1,
            i,
            conv_config,
            device,
            out_height,
            out_width,
            itr=1,
            landmark=True,
        )
    print("backbone 1 completed")

    for i in range(0, 4):
        if i == 0:
            y, out_height, out_width = blazeblock(
                x,
                24,
                48,
                3,
                2,
                0,
                False,
                parameters.backbone2,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=2,
                landmark=True,
            )
        else:
            y, out_height, out_width = blazeblock(
                y,
                48,
                48,
                3,
                1,
                1,
                False,
                parameters.backbone2,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=2,
                landmark=True,
            )
    print("backbone 2 completed")

    for i in range(0, 5):
        if i == 0:
            z, out_height, out_width = blazeblock(
                y,
                48,
                96,
                3,
                2,
                0,
                False,
                parameters.backbone3,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=3,
                landmark=True,
            )
        else:
            z, out_height, out_width = blazeblock(
                z,
                96,
                96,
                3,
                1,
                1,
                False,
                parameters.backbone3,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=3,
                landmark=True,
            )
    print("backbone 3 completed")

    for i in range(0, 6):
        if i == 0:
            w, out_height, out_width = blazeblock(
                z,
                96,
                192,
                3,
                2,
                0,
                False,
                parameters.backbone4,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=4,
                landmark=True,
            )
        else:
            w, out_height, out_width = blazeblock(
                w,
                192,
                192,
                3,
                1,
                1,
                False,
                parameters.backbone4,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=4,
                landmark=True,
            )
    print("backbone 4 completed")

    for i in range(0, 7):
        if i == 0:
            v, out_height, out_width = blazeblock(
                w,
                192,
                288,
                3,
                2,
                0,
                False,
                parameters.backbone5,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=5,
                landmark=True,
            )
        else:
            v, out_height, out_width = blazeblock(
                v,
                288,
                288,
                3,
                1,
                1,
                False,
                parameters.backbone5,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                itr=5,
                landmark=True,
            )
    print("backbone 5 completed")
    print("output shape from blazepose block 5 passed to ttnn conv", v.shape)
    print("==============")

    print("TTNN CONV up1 [0]")
    bias = ttnn.from_torch(parameters.up1[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    weight = ttnn.from_torch(parameters.up1[0].weight, dtype=ttnn.bfloat16)
    print("input shape", v.shape, out_height, out_width)
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    [v1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=v,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=288,
        out_channels=288,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=v.shape[0],
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=288,
    )
    print("output shape", v1.shape, out_height, out_width)
    print("==========")

    if torch_conv:
        v1 = ttnn.to_torch(v1).to(torch.float)
        v1 = torch.permute(v1, (0, 3, 1, 2))
        v1 = torch.reshape(v1, (1, v1.shape[1], out_height, out_width))
        up_conv = nn.Conv2d(288, 48, 1)
        up_conv.weight = parameters.up1[1].weight
        up_conv.bias = parameters.up1[1].bias
        print("TORCH CONV up1 [1]")
        print("input shape of v1", v1.shape)
        v1 = up_conv(v1)
        print("output shape of v1", v1.shape)
        out_height = v1.shape[2]
        out_width = v1.shape[3]
        v1 = torch.reshape(v1, (1, 1, v1.shape[1], out_height * out_width))
        v1 = ttnn.from_torch(v1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        v1 = ttnn.permute(v1, (0, 1, 3, 2))

    else:
        print("TTNN CONV up1 [1]")
        weight = ttnn.from_torch(parameters.up1[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up1[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("weight", weight.shape)
        print("bias", bias.shape)
        print("input shape", v1.shape, out_width, out_height)
        [v1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=v1,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=288,
            out_channels=48,
            device=device,
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

    v1 = ttnn.relu(v1)
    print("==========")

    print("TTNN CONV up2 [0]")
    weight = ttnn.from_torch(parameters.up2[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up2[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    out_height = w.shape[-3]
    out_width = w.shape[-2]
    print("input shape", w.shape, out_height, out_width)
    [w1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=w,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=192,
        out_channels=192,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=192,
    )
    print("output shape", w1.shape, out_height, out_width)
    print("==========")

    if torch_conv:
        w1 = ttnn.to_torch(w1).to(torch.float)
        w1 = torch.permute(w1, (0, 3, 1, 2))
        w1 = torch.reshape(w1, (1, w1.shape[1], out_height, out_width))
        up2_conv = nn.Conv2d(192, 48, 1)
        up2_conv.weight = parameters.up2[1].weight
        up2_conv.bias = parameters.up2[1].bias
        print("TORCH CONV up2 [1]")
        print("input shape of w1", w1.shape)
        w1 = up2_conv(w1)
        print("output shape of w1", w1.shape)
        print("==========")
        out_height = w1.shape[2]
        out_width = w1.shape[3]
        w1 = torch.reshape(w1, (1, 1, w1.shape[1], out_height * out_width))
        w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        w1 = ttnn.permute(w1, (0, 1, 3, 2))
    else:
        print("TTNN CONV up2 [1]")
        weight = ttnn.from_torch(parameters.up2[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up2[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("input shape", w1.shape)
        out_height = 16
        out_width = 16
        [w1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=w1,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=192,
            out_channels=48,
            device=device,
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

    w1 = ttnn.relu(w1)

    v1 = ttnn.to_layout(v1, ttnn.ROW_MAJOR_LAYOUT)

    w1 = ttnn.to_torch(w1)
    v1 = ttnn.to_torch(v1)
    w1 = torch.permute(w1, (0, 3, 1, 2))
    w1 = torch.reshape(w1, (1, w1.shape[1], out_height, out_width))

    v1 = torch.permute(v1, (0, 3, 1, 2))
    v1 = torch.reshape(v1, (1, v1.shape[1], 8, 8))

    w1 = w1 + F.interpolate(v1, scale_factor=2, mode="bilinear")

    w1 = torch.reshape(w1, (1, 1, w1.shape[1], out_height * out_width))
    w1 = torch.permute(w1, (0, 1, 3, 2))
    w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    v1 = torch.reshape(v1, (1, 1, v1.shape[1], 8 * 8))
    v1 = ttnn.from_torch(v1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    v1 = ttnn.permute(v1, (0, 1, 3, 2))

    print("TTNN CONV up3[0]")
    print("input shape", z.shape, out_height, out_width)
    weight = ttnn.from_torch(parameters.up3[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up3[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    out_height = z.shape[-3]
    out_width = z.shape[-2]
    [z1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=z,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=96,
        out_channels=96,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=96,
    )
    print("output shape", z1.shape, out_height, out_width)
    print("==========")

    if torch_conv:
        print("TORCH CONV up3[1]")
        z1 = ttnn.to_torch(z1).to(torch.float)
        z1 = torch.permute(z1, (0, 3, 1, 2))
        z1 = torch.reshape(z1, (1, z1.shape[1], out_height, out_width))
        up3_conv = nn.Conv2d(192, 48, 1)
        up3_conv.weight = parameters.up3[1].weight
        up3_conv.bias = parameters.up3[1].bias
        print("input shape of z1", z1.shape)
        z1 = up3_conv(z1)
        print("output shape of z1", z1.shape)
        out_height = z1.shape[2]
        out_width = z1.shape[3]
        z1 = torch.reshape(z1, (1, 1, z1.shape[1], out_height * out_width))
        z1 = ttnn.from_torch(z1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        z1 = ttnn.permute(z1, (0, 1, 3, 2))
    else:
        print("TTNN CONV up3[1]")
        print("input shape", z1.shape, out_height, out_width)
        weight = ttnn.from_torch(parameters.up3[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up3[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        [z1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=z1,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=96,
            out_channels=96,
            device=device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=96,
        )
        print("output shape", z1.shape, out_height, out_width)
        print("==========")

    z1 = ttnn.relu(z1)

    z1 = ttnn.to_torch(z1)
    w1 = ttnn.to_torch(w1)

    z1 = torch.permute(z1, (0, 3, 1, 2))
    z1 = torch.reshape(z1, (1, z1.shape[1], out_height, out_width))

    w1 = torch.permute(w1, (0, 3, 1, 2))
    w1 = torch.reshape(w1, (1, w1.shape[1], 16, 16))

    z1 = z1 + F.interpolate(w1, scale_factor=2, mode="bilinear")
    # z1 = ttnn.from_torch(z1, device = device, dtype = ttnn.bfloat16, layout = ttnn.TILE_LAYOUT)

    z1 = torch.reshape(z1, (1, 1, z1.shape[1], out_height * out_width))

    z1 = torch.permute(z1, (0, 1, 3, 2))
    z1 = ttnn.from_torch(z1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w1 = torch.reshape(w1, (1, 1, w1.shape[1], 16 * 16))
    w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    w1 = ttnn.permute(w1, (0, 1, 3, 2))

    weight = ttnn.from_torch(parameters.up4[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up4[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up4[0]")
    out_height = y.shape[-3]
    out_width = y.shape[-2]
    print("input shape", y.shape, out_height, out_width)
    [y1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=y,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=48,
        out_channels=48,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=48,
    )
    print("output shape", y1.shape, out_height, out_width)
    print("===========")

    if torch_conv:
        y1 = ttnn.to_torch(y1).to(torch.float)
        y1 = torch.permute(y1, (0, 3, 1, 2))
        y1 = torch.reshape(y1, (1, y1.shape[1], out_height, out_width))
        up4_conv = nn.Conv2d(48, 48, 1)
        up4_conv.weight = parameters.up4[1].weight
        up4_conv.bias = parameters.up4[1].bias
        print("TORCH CONV up4 [1]")
        print("input shape of y1", y1.shape)
        y1 = up4_conv(y1)
        print("output shape of y1", y1.shape)
        out_height = y1.shape[2]
        out_width = y1.shape[3]
        y1 = torch.reshape(y1, (1, 1, y1.shape[1], out_height * out_width))
        y1 = ttnn.from_torch(y1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        y1 = ttnn.permute(y1, (0, 1, 3, 2))
    else:
        weight = ttnn.from_torch(parameters.up4[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up4[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        [y1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=y1,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=48,
            out_channels=48,
            device=device,
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

    y1 = ttnn.relu(y1)

    y1 = ttnn.to_torch(y1)
    z1 = ttnn.to_torch(z1)

    y1 = torch.permute(y1, (0, 3, 1, 2))
    y1 = torch.reshape(y1, (1, y1.shape[1], out_height, out_width))

    z1 = torch.permute(z1, (0, 3, 1, 2))
    z1 = torch.reshape(z1, (1, z1.shape[1], 32, 32))

    y1 = y1 + F.interpolate(z1, scale_factor=2, mode="bilinear")

    y1 = torch.reshape(y1, (1, 1, y1.shape[1], out_height * out_width))
    y1 = torch.permute(y1, (0, 1, 3, 2))
    y1 = ttnn.from_torch(y1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    z1 = torch.reshape(z1, (1, 1, z1.shape[1], 32 * 32))
    z1 = ttnn.from_torch(z1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    z1 = ttnn.permute(z1, (0, 1, 3, 2))

    weight = ttnn.from_torch(parameters.up9[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up9[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up9[0]")
    print("input shape", x.shape, out_height, out_width)
    out_height = x.shape[-3]
    out_width = x.shape[-2]
    [x1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=24,
        out_channels=24,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=24,
    )
    print("output shape", x1.shape)
    print("=======")

    if torch_conv:
        x1 = ttnn.to_torch(x1).to(torch.float)
        x1 = torch.permute(x1, (0, 3, 1, 2))
        x1 = torch.reshape(x1, (1, x1.shape[1], out_height, out_width))
        up9_conv = nn.Conv2d(24, 8, 1)
        up9_conv.weight = parameters.up9[1].weight
        up9_conv.bias = parameters.up9[1].bias
        print("TORCH CONV up9[1]")
        print("input shape of x1", x1.shape)
        x1 = up9_conv(x1)
        print("output shape of x1", x1.shape)
        out_height = x1.shape[2]
        out_width = x1.shape[3]
        x1 = torch.reshape(x1, (1, 1, x1.shape[1], out_height * out_width))
        x1 = torch.permute(x1, (0, 1, 3, 2))
        x1 = ttnn.from_torch(x1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        print("=======-----=======")
    else:
        weight = ttnn.from_torch(parameters.up9[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up9[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("input shape up9[1]", x1.shape, out_height, out_width)
        [x1, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=x1,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=24,
            out_channels=48,
            device=device,
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
        print("output shape up9[1]", x1.shape)
        print("=======-----=======")
    x1 = ttnn.relu(x1)

    weight = ttnn.from_torch(parameters.up8[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up8[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up8[0]")
    print("input shape", y1.shape, out_height, out_width)
    [conv8, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=y1,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=48,
        out_channels=48,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=64,
        input_width=64,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=48,
    )
    print("conv8 up8[0]", conv8.shape)
    print("=======-----=======")

    if torch_conv:
        conv8 = ttnn.to_torch(conv8).to(torch.float)
        conv8 = torch.permute(conv8, (0, 3, 1, 2))
        conv8 = torch.reshape(conv8, (1, conv8.shape[1], out_height, out_width))
        up8_conv = nn.Conv2d(48, 8, 1)
        up8_conv.weight = parameters.up8[1].weight
        up8_conv.bias = parameters.up8[1].bias
        print("TORCH CONV up8[1]")
        print("input shape of conv8", conv8.shape, out_height, out_width)
        conv8 = up8_conv(conv8)
        out_height = conv8.shape[2]
        out_width = conv8.shape[3]
        conv8 = torch.reshape(conv8, (1, 1, conv8.shape[1], out_height * out_width))
        conv8 = torch.permute(conv8, (0, 1, 3, 2))
        conv8 = ttnn.from_torch(conv8, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        print("output shape of conv8", conv8.shape)
        print("=======-----=======")
    else:
        weight = ttnn.from_torch(parameters.up8[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.up8[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        [conv8, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=conv8,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=48,
            out_channels=8,
            device=device,
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

    conv8 = ttnn.relu(conv8)

    conv8 = ttnn.to_torch(conv8)
    x1 = ttnn.to_torch(x1)

    conv8 = torch.permute(conv8, (0, 3, 1, 2))
    conv8 = torch.reshape(conv8, (1, conv8.shape[1], out_height, out_width))

    x1 = torch.permute(x1, (0, 3, 1, 2))
    x1 = torch.reshape(x1, (1, x1.shape[1], 128, 128))

    seg = x1 + F.interpolate(conv8, scale_factor=2, mode="bilinear")

    conv8 = torch.reshape(conv8, (1, 1, conv8.shape[1], out_height * out_width))
    conv8 = ttnn.from_torch(conv8, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    conv8 = ttnn.permute(conv8, (0, 1, 3, 2))

    x1 = torch.reshape(x1, (1, 1, x1.shape[1], 128 * 128))
    x1 = ttnn.from_torch(x1, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    x1 = ttnn.permute(x1, (0, 1, 3, 2))

    seg = torch.reshape(seg, (1, 1, seg.shape[1], 128 * 128))
    seg = torch.permute(seg, (0, 1, 3, 2))
    seg = ttnn.from_torch(seg, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    print("=======-----=======")

    weight = ttnn.from_torch(parameters.block6[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.block6[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    out_height = 128
    out_width = 128
    print("TTNN CONV block6[0]", seg.shape, out_height, out_width)
    [block6, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=seg,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=8,
        out_channels=8,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=8,
    )
    print("output shape block6[0]", block6.shape)
    print("=======-----=======")

    if torch_conv:
        block6 = ttnn.to_torch(block6).to(torch.float)
        block6 = torch.permute(block6, (0, 3, 1, 2))
        block6 = torch.reshape(block6, (1, block6.shape[1], out_height, out_width))
        block6_conv = nn.Conv2d(8, 8, 1)
        block6_conv.weight = parameters.block6[1].weight
        block6_conv.bias = parameters.block6[1].bias
        print("TORCH CONV block6[1]")
        print("input shape of block6[1]", block6.shape, out_height, out_width)
        block6 = block6_conv(block6)
        out_height = block6.shape[2]
        out_width = block6.shape[3]
        block6 = torch.reshape(block6, (1, 1, block6.shape[1], out_height * out_width))
        block6 = torch.permute(block6, (0, 1, 3, 2))
        block6 = ttnn.from_torch(block6, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        print("output shape of block6[1]", block6.shape)
        print("=======-----=======")
    else:
        weight = ttnn.from_torch(parameters.block6[1].weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.block6[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        [block6, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=block6,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=8,
            out_channels=8,
            device=device,
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

    seg = ttnn.relu(block6)
    if False:  # torch_conv:
        print("input shape", seg.shape, out_height, out_width)
        seg = ttnn.to_torch(seg).to(torch.float)
        seg = torch.permute(seg, (0, 3, 1, 2))
        seg = torch.reshape(seg, (1, seg.shape[1], out_height, out_width))
        segmentation = nn.Conv2d(8, 1, 3, padding=1)
        segmentation.weight = parameters.segmentation.weight
        segmentation.bias = parameters.segmentation.bias
        print("TORCH CONV SEGMENTATION")
        print("input shape of seg", seg.shape)
        seg = segmentation(seg)
        print("output shape of seg", seg.shape)
        print("=======-----=======")
        # seg = ttnn.from_torch(seg, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        seg = torch.reshape(seg, (1, 1, seg.shape[1], out_height * out_width))
        seg = torch.permute(seg, (0, 1, 3, 2))
        seg = ttnn.from_torch(seg, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        print("TTNN CONV SEGMENTATION")
        print("input shape segementation", seg.shape)
        weight = ttnn.from_torch(parameters.segmentation.weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.segmentation.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        [seg, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=seg,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=8,
            out_channels=1,
            device=device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=1,
            input_height=out_height,
            input_width=out_width,
            conv_config=conv_config,
            conv_op_cache={},
            debug=False,
            groups=1,
        )
    seg = ttnn.to_torch(seg).squeeze(1)
    print("output shape of segmentation", seg.shape)
    print("=======-----=======")
    seg = ttnn.from_torch(seg, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    weight = ttnn.from_torch(parameters.up5[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up5[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    out_height = z.shape[-3]
    out_width = z.shape[-2]
    print("TTNN CONV up5[0]", z.shape, out_height, out_width)
    [up5, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=z,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=96,
        out_channels=96,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=96,
    )
    print("output shape of up5[0]", up5.shape)
    print("=======-----=======")

    weight = ttnn.from_torch(parameters.up5[1].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up5[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up5[1]")
    print("input shape", up5.shape)
    [up5, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=up5,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=96,
        out_channels=96,
        device=device,
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
    print("output shape of up5[1]", up5.shape)
    print("=======-----=======")

    up5 = ttnn.relu(up5)

    y1 = ttnn.reshape(y1, (y1.shape[0], 64, 64, y1.shape[-1]))

    for i in range(0, 5):
        if i == 0:
            out, out_height, out_width = blazeblock(
                y1,
                48,
                96,
                3,
                2,
                0,
                False,
                parameters.block1,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=6,
            )
        else:
            out, out_height, out_width = blazeblock(
                out,
                96,
                96,
                3,
                1,
                1,
                False,
                parameters.block1,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=6,
            )
    print("blazepose block 1 completed")

    # up5 = ttnn.to_layout(up5, ttnn.TILE_LAYOUT)
    # out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    print("up5 shape in adddd", up5.shape)
    print("out shape in adddd", out.shape)
    up5 = ttnn.reshape(up5, (up5.shape[0], 32, 32, up5.shape[-1]))
    out = ttnn.to_torch(out) + ttnn.to_torch(up5)
    out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # out = ttnn.add(out, up5)

    weight = ttnn.from_torch(parameters.up6[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up6[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up6[0]")
    print("input shape", w.shape, out_height, out_width)
    [up6, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=w,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=192,
        out_channels=192,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=w.shape[-3],
        input_width=w.shape[-2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=192,
    )
    print("output shape", up6.shape)
    print("=======-----=======")

    weight = ttnn.from_torch(parameters.up6[1].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up6[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up6[1]")
    print("input shape", up6.shape, out_height, out_width)
    [up6, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=up6,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=192,
        out_channels=192,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=1,
        input_height=16,
        input_width=16,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )
    print("output shape", up6.shape)
    print("=======-----=======")

    up6 = ttnn.relu(up6)

    for i in range(0, 6):
        if i == 0:
            out, out_height, out_width = blazeblock(
                out,
                96,
                192,
                3,
                2,
                0,
                False,
                parameters.block2,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=7,
            )
        else:
            out, out_height, out_width = blazeblock(
                out,
                192,
                192,
                3,
                1,
                1,
                False,
                parameters.block2,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=7,
            )
    print("blazepose block 2 completed")

    up6 = ttnn.to_layout(up6, ttnn.ROW_MAJOR_LAYOUT)
    print("up6 shape in adddd", up6.shape)
    print("out shape in adddd", out.shape)
    up6 = ttnn.reshape(up6, (up6.shape[0], 16, 16, up6.shape[-1]))
    # out = out + up6
    # up6 = ttnn.to_layout(up6, ttnn.TILE_LAYOUT)
    # out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    # out = ttnn.add(out, up6)
    out = ttnn.to_torch(out) + ttnn.to_torch(up6)
    out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    weight = ttnn.from_torch(parameters.up7[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up7[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up7[0]")
    out_height = v.shape[-3]
    out_width = v.shape[-2]
    print("input shape", v.shape, out_height, out_width)
    [up7, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=v,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=288,
        out_channels=288,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=288,
    )
    print("output shape", up7.shape)
    print("=======-----=======")

    weight = ttnn.from_torch(parameters.up7[1].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.up7[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV up7[1]")
    print("input shape", up7.shape)
    [up7, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=up7,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=288,
        out_channels=288,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=1,
        input_height=up7.shape[-3],
        input_width=up7.shape[-2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )
    print("output shape", up7.shape)
    print("=======-----=======")

    up7 = ttnn.relu(up7)

    for i in range(0, 7):
        if i == 0:
            out, out_height, out_width = blazeblock(
                out,
                192,
                288,
                3,
                2,
                0,
                False,
                parameters.block3,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=8,
            )
        else:
            out, out_height, out_width = blazeblock(
                out,
                288,
                288,
                3,
                1,
                1,
                False,
                parameters.block3,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=8,
            )
    print("blazepose block 3 completed")

    up7 = ttnn.to_layout(up7, ttnn.ROW_MAJOR_LAYOUT)
    print("up7 shape in adddd", up7.shape)
    print("out shape in adddd", out.shape)
    up7 = ttnn.reshape(up7, (up7.shape[0], 8, 8, up7.shape[-1]))
    out = ttnn.to_torch(out) + ttnn.to_torch(up7)
    # out = out + up7
    out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    for i in range(0, 15):
        if i == 0 or i == 8:
            out, out_height, out_width = blazeblock(
                out,
                288,
                288,
                3,
                2,
                0,
                False,
                parameters.block4,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=9,
            )
        else:
            out, out_height, out_width = blazeblock(
                out,
                288,
                288,
                3,
                1,
                1,
                False,
                parameters.block4,
                i,
                conv_config,
                device,
                out_height,
                out_width,
                landmark=True,
                itr=9,
            )
    print("blazepose block 4 completed")
    print("outout", out.layout)

    temp = out

    out = ttnn.to_torch(out)
    print("out before pad", out.shape)
    out = F.pad(out, (0, 0, 0, 0, 0, 0), "constant", 0)
    print("out after pad", out.shape)
    out = ttnn.from_torch(out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    weight = ttnn.from_torch(parameters.block5.convs[0].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.block5.convs[0].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV block5 conv[0]")
    print("input shape", temp.shape)
    temp = ttnn.to_layout(temp, ttnn.ROW_MAJOR_LAYOUT)
    [temp, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=temp,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=288,
        out_channels=288,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=1,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )
    print("output shape", temp.shape)
    print("=======-----=======")

    weight = ttnn.from_torch(parameters.block5.convs[1].weight, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(parameters.block5.convs[1].bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
    print("TTNN CONV block5 conv[1]")
    print("input shape", temp.shape, temp.layout)
    temp = ttnn.to_layout(temp, ttnn.ROW_MAJOR_LAYOUT)
    [temp, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=temp,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=288,
        out_channels=288,
        device=device,
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
    print("output shape", temp.shape)
    print("=======-----=======")

    temp = ttnn.to_layout(temp, ttnn.ROW_MAJOR_LAYOUT)
    temp = ttnn.reshape(temp, (temp.shape[0], 2, 2, temp.shape[-1]))
    out = ttnn.relu(out + temp)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)

    if torch_conv:
        print("input shape", out.shape, out_height, out_width)
        out = ttnn.to_torch(out).to(torch.float)
        out = torch.permute(out, (0, 3, 1, 2))
        out = torch.reshape(out, (1, out.shape[1], out_height, out_width))
        flag = nn.Conv2d(288, 1, 2)
        flag.weight = parameters.flag.weight
        flag.bias = parameters.flag.bias
        print("TORCH CONV flag")
        print("input shape of out", out.shape)
        flag_out = flag(out)
        print("output shape of flag_out", flag_out.shape)
        flag_out = ttnn.from_torch(flag_out, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        out = torch.reshape(out, (1, 1, out.shape[1], out_height * out_width))
        out = torch.permute(out, (0, 1, 3, 2))
        out = ttnn.from_torch(out, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        weight = ttnn.from_torch(parameters.flag.weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.flag.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("TTNN CONV flag")
        print("input shape", out.shape, out_height, out_width)
        [flag_out, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=288,
            out_channels=1,
            device=device,
            kernel_size=(2, 2),
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
        print("output shape", flag_out.shape)
        print("=======-----=======")

    flag_out = ttnn.to_torch(flag_out).view(-1).sigmoid()
    flag_out = ttnn.from_torch(flag_out, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # print("flag_out before sigmoid",flag_out.shape)
    # flag_out = ttnn.sigmoid(flag_out)
    # print("flag_out",flag_out.shape)
    print("======")
    if False:  # torch_conv:
        print("input shape", out.shape, out_height, out_width)
        out = ttnn.to_torch(out).to(torch.float)
        out = torch.permute(out, (0, 3, 1, 2))
        out = torch.reshape(out, (1, out.shape[1], out_height, out_width))
        landmarks = nn.Conv2d(288, 124, 2)
        landmarks.weight = parameters.landmarks.weight
        landmarks.bias = parameters.landmarks.bias
        print("TORCH CONV landmarks")
        print("input shape of out", out.shape)
        landmark = landmarks(out)
        print("output shape of landmark", landmark.shape)
        # landmark = ttnn.from_torch(landmark, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        landmark = torch.reshape(landmark, (1, 1, 1, landmark.shape[-3]))
        landmark = torch.permute(landmark, (0, 1, 3, 2))
        print("output shape of landmark after permute", landmark.shape)
        landmarks = ttnn.from_torch(landmark, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        weight = ttnn.from_torch(parameters.landmarks.weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(parameters.landmarks.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)
        print("TTNN CONV landmarks")
        [landmarks, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=288,
            out_channels=124,
            device=device,
            kernel_size=(2, 2),
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
    landmark = ttnn.to_torch(landmarks).view(batch, 31, 4) / 256
    landmark = ttnn.from_torch(landmark, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("output shape landmark", landmark.shape)
    # print("seg",seg)
    seg = ttnn.to_torch(seg)
    # seg = torch.permute(seg,(0, 2, 1))
    seg = torch.reshape(seg, (1, 128, 128))
    seg = ttnn.from_torch(seg, dtype=ttnn.bfloat16)
    return flag_out, landmark, seg


def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    resolution = 256
    points = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]], device=scale.device).view(1, 2, 4)
    points = points * scale.view(-1, 1, 1) / 2
    theta = theta.view(-1, 1, 1)
    R = torch.cat(
        (
            torch.cat((torch.cos(theta), -torch.sin(theta)), 2),
            torch.cat((torch.sin(theta), torch.cos(theta)), 2),
        ),
        1,
    )
    center = torch.cat((xc.view(-1, 1, 1), yc.view(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res - 1], [0, res - 1, 0]], dtype=np.float32).T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].cpu().numpy().T
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res))  # , borderValue=127.5)
        img = torch.tensor(img, device=scale.device)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype("float32")
        affine = torch.tensor(affine, device=scale.device)
        affines.append(affine)
    if imgs:
        imgs = torch.stack(imgs).permute(0, 3, 1, 2).float() / 255.0  # / 127.5 - 1.0
        affines = torch.stack(affines)
    else:
        imgs = torch.zeros((0, 3, res, res), device=scale.device)
        affines = torch.zeros((0, 2, 3), device=scale.device)

    return imgs, affines, points


def denormalize_landmarks(landmarks, affines):
    resolution = 256
    landmarks[:, :, :2] *= resolution
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:, :2] @ landmark[:, :2].T + affine[:, 2:]).T
        landmarks[i, :, :2] = landmark
    return landmarks
