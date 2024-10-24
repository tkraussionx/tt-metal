# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
from torch import nn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


class Conv:
    def __init__(
        self,
        model,
        input_params,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        fused_op=True,
        width_sharding=False,
    ) -> None:
        weight, bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        self.weights = ttnn.from_torch(weight)
        bias = bias.reshape(1, 1, 1, -1)
        self.bias = ttnn.from_torch(bias)
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.deallocate = deallocate
        self.activation = activation

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=self.shard_layout,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            act_block_w_div=1,
            packer_l1_accum_enabled=False,
            input_channels_alignment=16 if self.input_params[3] < 16 else 32,
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
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
        )
        return output_tensor


class Sample(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)

    def forward(self, input1):
        x3 = self.c3(input1)
        x3_b = self.b3(x3)
        x3_m = self.relu(x3_b)

        # maxpools
        x4 = self.p1(x3_m)
        return x3_m, x4


class TtSample:
    def __init__(self, model) -> None:
        print("model", model)
        self.conv3 = Conv(
            model,
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=True,
            height_sharding=False,  # keeping it true doesn't affect PCC of maxpool
        )

    def __call__(self, device, input_tensor):
        output_tensor = self.conv3(device, input_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        pool_1 = ttnn.max_pool2d(
            input_tensor=output_tensor,
            batch_size=1,
            input_h=10,
            input_w=10,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        return output_tensor, pool_1


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_neck(device, reset_seeds):
    torch_model = Sample()

    torch_input_tensor1 = torch.randn(1, 10, 10, 1024, dtype=torch.bfloat16)

    ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
    ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 100, 1024))
    ttnn_input_tensor1 = ttnn.to_layout(ttnn_input_tensor1, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device=device)

    torch_input_tensor1 = torch_input_tensor1.permute(0, 3, 1, 2).float()
    torch_model.eval()
    ttnn_model = TtSample(torch_model)
    result_ttnn = ttnn_model(device, ttnn_input_tensor1)

    result_1 = ttnn.to_torch(result_ttnn[0])
    result_2 = ttnn.to_torch(result_ttnn[1])

    ref1, ref2 = torch_model(torch_input_tensor1)
    ref1 = ref1.permute(0, 2, 3, 1)
    ref2 = ref2.permute(0, 2, 3, 1)

    result1 = result_1.reshape(ref1.shape)
    result2 = result_2.reshape(ref2.shape)

    assert_with_pcc(result1, ref1, 0.99)  # PCC = 0.99
    assert_with_pcc(result2, ref2, 0.99)  # PCC = 0.055
