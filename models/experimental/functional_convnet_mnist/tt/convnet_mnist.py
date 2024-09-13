# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import torch.nn.functional as F
from torch import nn


def functional_convnet_mnist(
    input_tensor,
    parameters,
    device,
):
    batch_size = input_tensor.shape[0]

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

    x = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.conv1.weight,
        in_channels=1,
        out_channels=32,
        device=device,
        bias_tensor=parameters.conv1.bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=True,
        groups=1,
    )
    x = ttnn.relu(x)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=30,
        input_w=30,
        channels=32,
        kernel_size=[2, 2],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        device=device,
    )

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.conv2.weight,
        in_channels=32,
        out_channels=64,
        device=device,
        bias_tensor=parameters.conv2.bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=15,
        input_width=15,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    x = ttnn.relu(x)

    x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[2, 2],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        device=device,
    )

    x = ttnn.from_device(x)
    x = ttnn.reshape(x, (x.shape[0], -1))

    fc1_weight = parameters.fc1.weight
    fc1_bias = parameters.fc1.bias
    fc1_weight = ttnn.to_device(fc1_weight, device=device)
    fc1_bias = ttnn.to_device(fc1_bias, device=device)
    x = ttnn.to_device(x, device)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = x @ fc1_weight
    x = x + fc1_bias
    x = ttnn.relu(x)

    fc2_weight = ttnn.to_device(parameters.fc2.weight, device=device)
    fc2_bias = ttnn.to_device(parameters.fc2.bias, device=device)
    x = x @ fc2_weight
    x = x + fc2_bias

    output = torch.softmax(ttnn.to_torch(x), dim=-1)
    output = ttnn.from_torch(output, device=device, dtype=ttnn.bfloat16)
    return output


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, device):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        weight = model.weight
        bias = model.bias
        while weight.dim() < 4:
            weight = weight.unsqueeze(0)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)

    return parameters
