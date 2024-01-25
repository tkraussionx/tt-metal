# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import tt_lib
from typing import Optional


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def resnetBlock2D(
    input_tensor,
    *,
    temb,
    in_channels,
    parameters,
    device,
    temb_channels=1280,
    groups: int = 32,
    time_embedding_norm: str = "default",
    output_scale_factor: float = 1.0,
    out_channels: Optional[int] = None,
    non_linearity="silu",
    pre_norm=True,
    eps=1e-5,
    up=False,
    down=False,
    use_in_shortcut: Optional[bool] = None,
):
    if non_linearity == "mish":
        assert False, "Mish is not implemented!"
    else:
        nonlinearity = ttnn.silu

    out_channels = in_channels if out_channels is None else out_channels
    hidden_states = input_tensor
    hidden_states = ttnn.group_norm(
        hidden_states, num_groups=groups, weight=parameters.norm1.weight, bias=parameters.norm1.bias, epsilon=eps
    )
    hidden_states = nonlinearity(hidden_states)

    if up:
        assert False, "Up block within residual block is not implemented!"
    elif down:
        assert False, "Down block within residual block is not implemented"

    # Using fallback Conv2D as we face issue with ttnn.Conv2D
    conv1 = fallback_ops.Conv2d(
        parameters.conv1.weight,
        parameters.conv1.bias,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    hidden_states = ttnn_to_torch(hidden_states)
    hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
    hidden_states = conv1(hidden_states)
    hidden_states = tt_to_torch_tensor(hidden_states)
    hidden_states = torch_to_ttnn(hidden_states, device=device)

    if temb is not None:
        temb = nonlinearity(temb)
        if temb_channels is not None:
            if time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {time_embedding_norm} ")
            # temb=ttnn.linear(temb,parameters.time_emb_proj.weight,bias=parameters.time_emb_proj.bias)
            temb = ttnn.matmul(temb, parameters.time_emb_proj.weight)
            temb = ttnn.add(temb, parameters.time_emb_proj.bias)

        temb = ttnn_to_torch(temb)
        temb = torch_to_tt_tensor_rm(temb, device, put_on_device=False)
        temb = tt_lib.tensor.permute(temb, (2, 3, 0, 1))

    if temb is not None and time_embedding_norm == "default":
        # Using tt_lib.tensor.bcast as we face issue with ttnn addition
        hidden_states = ttnn_to_torch(hidden_states)
        hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
        hidden_states = tt_lib.tensor.bcast(
            hidden_states,
            temb,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.HW,
        )
        hidden_states = tt_to_torch_tensor(hidden_states)
        hidden_states = torch_to_ttnn(hidden_states, device)

    hidden_states = ttnn.group_norm(
        hidden_states, num_groups=groups, weight=parameters.norm2.weight, bias=parameters.norm2.bias, epsilon=eps
    )
    hidden_states = nonlinearity(hidden_states)

    # Using fallback Conv2D as we face issue with ttnn.Conv2D
    conv2 = fallback_ops.Conv2d(
        parameters.conv2.weight,
        parameters.conv2.bias,
        out_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    hidden_states = ttnn_to_torch(hidden_states)
    hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
    hidden_states = conv2(hidden_states)
    hidden_states = tt_to_torch_tensor(hidden_states)
    hidden_states = torch_to_ttnn(hidden_states, device=device)

    use_in_shortcut = in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
    if use_in_shortcut:
        # Using fallback Conv2D as we face issue with ttnn.Conv2D
        conv_shortcut = fallback_ops.Conv2d(
            parameters.conv_shortcut.weight,
            parameters.conv_shortcut.bias,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        input_tensor = ttnn_to_torch(input_tensor)
        input_tensor = torch_to_tt_tensor_rm(input_tensor, device)
        input_tensor = conv_shortcut(input_tensor)
        input_tensor = tt_to_torch_tensor(input_tensor)
        input_tensor = torch_to_ttnn(input_tensor, device=device)

    output_sc_recip = 1 / output_scale_factor
    output_tensor = ttnn.add(input_tensor, hidden_states)
    output_tensor = ttnn.mul(output_tensor, output_sc_recip)

    return output_tensor
