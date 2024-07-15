# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class Conv:
    def __init__(
        self,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="relu",
    ) -> None:
        self.kernel_size = (conv_params[5], conv_params[6])
        self.conv_params = conv_params
        self.out_channels = conv_params[4]
        self.weights = ttnn.from_torch(torch.randn(conv_params[4], conv_params[3], conv_params[5], conv_params[6]))
        self.bias = ttnn.from_torch(torch.randn(1, 1, 1, conv_params[4]))
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.height_sharding = height_sharding
        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            height_sharding=self.height_sharding,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            input_channels_alignment=16 if self.conv_params[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.conv_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[7], self.conv_params[8]),
            padding=(self.conv_params[9], self.conv_params[10]),
            batch_size=self.conv_params[0],
            input_height=self.conv_params[1],
            input_width=self.conv_params[2],
            conv_config=conv_config,
        )
        return output_tensor
