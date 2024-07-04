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
    itr=1,
):
    channel_pad = out_channel - in_channel
    if stride == 2:
        if kernel_size == 3:
            h = ttnn.to_torch(x)
            h = F.pad(h, (0, 2, 0, 2), "constant", 0)
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16)
        else:
            h = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2)))
            h = F.pad(h, (1, 2, 1, 2), "constant", 0)
            h = torch.permute(h, (0, 2, 3, 1))
            h = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            out_height = h.shape[-2]
            out_width = h.shape[-1]

        if itr == 1 and (i == 5 or i == 16):
            max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            x = ttnn.permute(x, (0, 3, 1, 2))
            x = ttnn.to_torch(x)
            x = max_pool(x)
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
            x = ttnn.permute(x, (0, 2, 3, 1))

        else:
            x_shape = x.shape
            x = ttnn.reshape(x, (x.shape[0], 1, x.shape[-3] * x.shape[-2], x.shape[-1]))
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
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.reshape(x, (x_shape[0], x_shape[1] // 2, x_shape[1] // 2, x_shape[3]))

    else:
        h = x
    out_height_x = out_height
    out_width_x = out_width
    if skip_proj:
        if i == 5:
            x = ttnn.to_torch(x).to(torch.float)
            x = torch.permute(x, (0, 3, 1, 2))
            skip_conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            skip_conv.weight = nn.Parameter(ttnn.to_torch(parameters[i].skip_proj.weight).to(torch.float))
            skip_conv.bias = nn.Parameter(
                ttnn.to_torch(parameters[i].skip_proj.bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
            )
            x = skip_conv(x)
            out_height_x = x.shape[-2]
            out_width_x = x.shape[-1]
            # PCC is reduced with ttnn reshape but irreproducable in unit test
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2] * x.shape[3]))
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            x = ttnn.permute(ttnn.to_device(x, device=device), (0, 1, 3, 2))

        else:
            [x, out_height_x, out_width_x, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=parameters[i].skip_proj.weight,
                in_channels=in_channel,
                out_channels=out_channel,
                device=device,
                bias_tensor=parameters[i].skip_proj.bias,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                conv_config=conv_config,
                conv_op_cache={},
                debug=None,
                groups=1,
            )

    elif channel_pad > 0:
        x = ttnn.pad(x, (0, 0, 0, 0), value=0)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT)

    [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters[i].convs[0].weight,
        in_channels=in_channel,
        out_channels=in_channel,
        device=device,
        bias_tensor=parameters[i].convs[0].bias,
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

    if i == 2 or i == 3 or i == 4 or i == 5:
        h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.reshape(h, (1, out_height, out_width, h.shape[-1]))

        # Shard shape must be tile sized
        h = ttnn.sharded_to_interleaved(h, ttnn.L1_MEMORY_CONFIG)
        h = ttnn.permute(h, (0, 3, 1, 2))
        h = ttnn.to_torch(h).to(torch.float)
        conv2 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv2.weight = nn.Parameter(ttnn.to_torch(parameters[i].convs[1].weight).to(torch.float))
        conv2.bias = nn.Parameter(
            ttnn.to_torch(parameters[i].convs[1].bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
        )
        h = conv2(h)
        out_height = h.shape[2]
        out_width = h.shape[3]

        h = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.reshape(h, (1, 1, h.shape[1], out_height * out_width))
        h = ttnn.permute(ttnn.to_device(h, device=device), (0, 1, 3, 2))

    else:
        [h, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=h,
            weight_tensor=parameters[i].convs[1].weight,
            in_channels=in_channel,
            out_channels=out_channel,
            device=device,
            bias_tensor=parameters[i].convs[1].bias,
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

    h = ttnn.to_layout(h, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    h = ttnn.reshape(h, (h.shape[0], out_height, out_width, h.shape[-1]))

    # this->output_mem_config.is_sharded()
    h = ttnn.to_torch(h)
    h = torch.permute(h, (0, 3, 1, 2))
    h = ttnn.from_torch(h, dtype=ttnn.bfloat16, device=device)
    h = ttnn.to_layout(h, layout=ttnn.TILE_LAYOUT)

    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(ttnn.from_device(x), (1, out_height_x, out_width_x, x.shape[-1]))
    x = ttnn.permute(ttnn.to_device(x, device=device), (0, 3, 1, 2))
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

    temp = h + x
    out = ttnn.permute(ttnn.relu(temp), (0, 2, 3, 1))
    return ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT), out_height, out_width


def blazepose(x, parameters, device):
    detection2roi_method = "alignment"
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    dscale = 1.5
    dy = 0.0
    b = x.shape[0]
    use_shallow_conv_variant = False
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        activation="relu",
        reallocate_halo_output=True,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.backbone1[0].weight,
        in_channels=3,
        out_channels=48,
        device=device,
        bias_tensor=parameters.backbone1[0].bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        batch_size=1,
        input_height=128,
        input_width=128,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=False,
        groups=1,
    )

    x = ttnn.to_torch(x)
    x = torch.reshape(x, (x.shape[0], out_height, out_width, x.shape[-1]))

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    in_channel = [48, 48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128]
    out_channel = [48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128]

    i = 2
    for i in range(2, 24):
        if i > 1:
            if i == 5 or i == 9 or i == 16:
                x, out_height, out_width = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    2,
                    0,
                    True,
                    parameters.backbone1,
                    i,
                    None,
                    device,
                    out_height,
                    out_width,
                    1,
                )

            else:
                x, out_height, out_width = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    1,
                    2,
                    False,
                    parameters.backbone1,
                    i,
                    None,
                    device,
                    out_height,
                    out_width,
                    1,
                )

        i += 1

    it = 0

    for i in range(6):
        if it == 0:
            h, out_height, out_width = blazeblock(
                x, 128, 256, 5, 2, 0, True, parameters.backbone2, i, None, device, out_height, out_width, 2
            )
        else:
            h, out_height, out_width = blazeblock(
                h, 256, 256, 5, 1, 2, False, parameters.backbone2, i, None, device, out_height, out_width, 2
            )
        it += 1

    # Channel problem
    class8 = nn.Conv2d(128, 2, 1)
    class8.weight = nn.Parameter(ttnn.to_torch(parameters.classifier_8.weight).to(torch.float))
    class8.bias = nn.Parameter(
        ttnn.to_torch(parameters.classifier_8.bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
    )

    temp = ttnn.permute(x, (0, 3, 1, 2))
    temp = ttnn.to_torch(temp).to(torch.float)

    c1 = class8(temp)
    c1 = ttnn.from_torch(c1, dtype=ttnn.bfloat16, device=device)
    c1 = ttnn.permute(c1, (0, 2, 3, 1))
    c1 = ttnn.to_torch(c1)
    c1 = c1.reshape(b, -1, 1)

    class16 = nn.Conv2d(256, 6, 1)
    class16.weight = nn.Parameter(ttnn.to_torch(parameters.classifier_16.weight).to(torch.float))
    class16.bias = nn.Parameter(
        ttnn.to_torch(parameters.classifier_16.bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
    )

    temp_h = ttnn.permute(h, (0, 3, 1, 2))
    temp_h = ttnn.to_torch(temp_h).to(torch.float)

    c2 = class16(temp_h)
    c2 = ttnn.from_torch(c2, dtype=ttnn.bfloat16, device=device)
    c2 = ttnn.permute(c2, (0, 2, 3, 1))
    c2 = ttnn.to_torch(c2).to(torch.float)
    c2 = c2.reshape(b, -1, 1)

    c1 = ttnn.from_torch(c1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c2 = ttnn.from_torch(c2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    c = ttnn.concat([c1, c2], dim=1)

    regressor_8 = nn.Conv2d(128, 24, 1)
    regressor_8.weight = nn.Parameter(ttnn.to_torch(parameters.regressor_8.weight).to(torch.float))
    regressor_8.bias = nn.Parameter(
        ttnn.to_torch(parameters.regressor_8.bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
    )

    x = ttnn.permute(x, (0, 3, 1, 2))
    x = ttnn.to_torch(x).to(torch.float)
    r1 = regressor_8(x)
    r1 = ttnn.from_torch(r1, dtype=ttnn.bfloat16, device=device)
    r1 = ttnn.permute(r1, (0, 2, 3, 1))
    r1 = ttnn.to_torch(r1)
    r1 = r1.reshape(b, -1, 12)
    r1 = ttnn.from_torch(r1, dtype=ttnn.bfloat16)  #

    regressor_16 = nn.Conv2d(256, 72, 1)
    regressor_16.weight = nn.Parameter(ttnn.to_torch(parameters.regressor_16.weight).to(torch.float))
    regressor_16.bias = nn.Parameter(
        ttnn.to_torch(parameters.regressor_16.bias).squeeze(0).squeeze(0).squeeze(0).to(torch.float)
    )

    h = ttnn.permute(h, (0, 3, 1, 2))
    h = ttnn.to_torch(h).to(torch.float)
    r2 = regressor_16(h)

    r2 = ttnn.from_torch(r2, dtype=ttnn.bfloat16, device=device)
    r2 = ttnn.permute(r2, (0, 2, 3, 1))
    r2 = ttnn.to_torch(r2)
    r2 = r2.reshape(b, -1, 12)
    r2 = ttnn.from_torch(r2, dtype=ttnn.bfloat16)  #
    r = ttnn.concat([r1, r2], dim=1)

    return [r, c]
