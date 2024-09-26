import torch
import ttnn
import pytest


class Conv:
    def __init__(
        self,
        input_params,
        conv_params,
        output_channels,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        activation="",
        groups=1,
        dtype=ttnn.bfloat8_b,
    ) -> None:
        weight = [output_channels, input_params[3] // groups, conv_params[0], conv_params[1]]
        bias = [1, 1, 1, output_channels]
        weight = torch.randn(weight)
        bias = torch.randn(bias)

        self.weights = ttnn.from_torch(weight)
        self.bias = ttnn.from_torch(bias)
        self.dtype = dtype

        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.height_sharding = height_sharding
        self.deallocate = deallocate
        self.activation = activation

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if self.height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            ),
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=True,
            packer_l1_accum_enabled=False,
            input_channels_alignment=32,
            transpose_shards=True,
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
            stride=(self.conv_params[2], self.conv_params[3]),
            padding=(self.conv_params[4], self.conv_params[5]),
            dilation=(self.conv_params[6], self.conv_params[7]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
        )
        return output_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_dilation(device):
    conv1_input_shape = [1, 32, 124, 124]
    conv1_weight_shape = [48, 32, 3, 3]
    conv1_bias_shape = [1, 1, 1, 48]
    torch_conv1_input_tensor_nchw = torch.randn(conv1_input_shape, dtype=torch.bfloat16).float()
    torch_conv1_weight_tensor = torch.randn(conv1_weight_shape, dtype=torch.bfloat16).float()
    torch_conv1_bias_tensor = torch.randn(conv1_bias_shape, dtype=torch.bfloat16).float()

    # Torch Conv1
    torch_conv1_output = torch.nn.functional.conv2d(
        torch_conv1_input_tensor_nchw,
        torch_conv1_weight_tensor,
        bias=torch_conv1_bias_tensor.reshape(-1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(2, 2),
        groups=1,
    )
    print("Torch Conv1 Output shape: ", torch_conv1_output.shape)

    conv2_weight_shape = [56, 48, 3, 3]
    conv2_bias_shape = [1, 1, 1, 56]
    torch_conv2_weight_tensor = torch.randn(conv2_weight_shape, dtype=torch.bfloat16).float()
    torch_conv2_bias_tensor = torch.randn(conv2_bias_shape, dtype=torch.bfloat16).float()

    # Torch Conv1
    torch_conv2_output = torch.nn.functional.conv2d(
        torch_conv1_output,
        torch_conv2_weight_tensor,
        bias=torch_conv2_bias_tensor.reshape(-1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(4, 4),
        groups=1,
    )
    print("Torch Conv2 Output shape: ", torch_conv2_output.shape)

    reader_patterns_cache = {}
    tt_conv1_weight_tensor = ttnn.from_torch(torch_conv1_weight_tensor, ttnn.bfloat16)
    tt_conv1_bias_tensor = ttnn.from_torch(torch_conv1_bias_tensor, ttnn.bfloat16)
    torch_input_tensor = torch.permute(torch_conv1_input_tensor_nchw, (0, 2, 3, 1))
    tt_conv1_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=True,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=True,
        reshard_if_not_optimal=False,
        deallocate_activation=False,
        reallocate_halo_output=False,
    )

    # TTNN Conv1
    [tt_conv1_output_tensor, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_conv1_input_tensor,
        weight_tensor=tt_conv1_weight_tensor,
        in_channels=32,
        out_channels=48,
        device=device,
        bias_tensor=tt_conv1_bias_tensor,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(2, 2),
        batch_size=1,
        input_height=124,
        input_width=124,
        conv_config=conv_config,
    )
    print("TTNN Conv1 Output shape: ", tt_conv1_output_tensor.shape)

    tt_conv2_weight_tensor = ttnn.from_torch(torch_conv2_weight_tensor, ttnn.bfloat16)
    tt_conv2_bias_tensor = ttnn.from_torch(torch_conv2_bias_tensor, ttnn.bfloat16)

    # TTNN Conv2
    [tt_conv2_output_tensor, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_conv1_output_tensor,
        weight_tensor=tt_conv2_weight_tensor,
        in_channels=48,
        out_channels=56,
        device=device,
        bias_tensor=tt_conv2_bias_tensor,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(4, 4),
        batch_size=1,
        input_height=120,
        input_width=120,
        conv_config=conv_config,
    )
    print("TTNN Conv2 Output shape: ", tt_conv2_output_tensor.shape)
    ttnn_output = ttnn.to_torch(tt_conv2_output_tensor)

    # ttnn_output = torch.reshape(ttnn_output, (1, 112, 112, 56)) # Expected shape as of torch Conv2 output
    # ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    # assert_with_pcc(ttnn_output, torch_conv2_output, pcc=0.99)
