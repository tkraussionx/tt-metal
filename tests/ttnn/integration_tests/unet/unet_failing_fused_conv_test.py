import torch
import torch.nn as nn
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, comp_allclose_and_pcc


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, UNet_dec_4_1):
            ttnn_module_args["decoder4_c1"] = ttnn_module_args.decoder4_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder4_1[0], model.decoder4_1[1])
            update_ttnn_module_args(ttnn_module_args["decoder4_c1"])
            ttnn_module_args["decoder4_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder4_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder4_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder4_c1"], decoder4_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder4_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_3_1):
            ttnn_module_args["decoder3_c1"] = ttnn_module_args.decoder3_1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder3_1[0], model.decoder3_1[1])
            update_ttnn_module_args(ttnn_module_args["decoder3_c1"])
            ttnn_module_args["decoder3_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c1"], decoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder3_c1"], return_parallel_config=True
            )
            return parameters

        if isinstance(model, UNet_dec_3_2):
            ttnn_module_args["decoder3_c2"] = ttnn_module_args.decoder3_2["0"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder3_2[0], model.decoder3_2[1])
            update_ttnn_module_args(ttnn_module_args["decoder3_c2"])
            ttnn_module_args["decoder3_c2"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c2"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c2"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c2"], decoder3_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder3_c2"], return_parallel_config=True
            )
            return parameters

    return custom_preprocessor


# decoder_4_1 model  start
class UNet_dec_4_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_4_1, self).__init__()

        features = init_features
        self.decoder4_1 = nn.Sequential(
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder4_1(x)
        return output


class TtUnet_dec_4_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec4_1 = parameters.decoder4_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec4_1.conv.input_sharded_memory_config)
        output_tensor_dec4_1 = self.dec4_1(input_tensor)

        return output_tensor_dec4_1


# decoder_4_1 model  end
# decoder_3_1 model  start
class UNet_dec_3_1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_3_1, self).__init__()

        features = init_features
        self.decoder3_1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder3_1(x)
        return output


class TtUnet_dec_3_1:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec3_1 = parameters.decoder3_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec3_1.conv.input_sharded_memory_config)
        output_tensor_dec3_1 = self.dec3_1(input_tensor)

        return output_tensor_dec3_1


# decoder_3_1 model  end


# decoder_3_2 model  start
class UNet_dec_3_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet_dec_3_2, self).__init__()

        features = init_features
        self.decoder3_2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.decoder3_2(x)
        return output


class TtUnet_dec_3_2:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.dec3_2 = parameters.decoder3_c2

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.dec3_2.conv.input_sharded_memory_config)
        output_tensor_dec3_2 = self.dec3_2(input_tensor)

        return output_tensor_dec3_2


# decoder_3_2 model  end


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_unet_dec_4_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder4.dec4conv1.weight",
                "decoder4.dec4norm1.weight",
                "decoder4.dec4norm1.bias",
                "decoder4.dec4norm1.running_mean",
                "decoder4.dec4norm1.running_var",
                "decoder4.dec4norm1.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_4_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 512, 60, 80)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_4_1(device, parameters, new_state_dict)

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

    output_tensor = output_tensor.reshape(1, 60, 80, 256)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder4_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.3347515936821286
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_unet_dec_3_1(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder3.dec3conv1.weight",
                "decoder3.dec3norm1.weight",
                "decoder3.dec3norm1.bias",
                "decoder3.dec3norm1.running_mean",
                "decoder3.dec3norm1.running_var",
                "decoder3.dec3norm1.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_3_1()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 256, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_3_1(device, parameters, new_state_dict)

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

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder3_1", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.44777801628917024
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_unet_dec_3_2(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            k
            in [
                "decoder3.dec3conv2.weight",
                "decoder3.dec3norm2.weight",
                "decoder3.dec3norm2.bias",
                "decoder3.dec3norm2.running_mean",
                "decoder3.dec3norm2.running_var",
                "decoder3.dec3norm2.num_batches_tracked",
            ]
        )
    }

    torch_model = UNet_dec_3_2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 120, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet_dec_3_2(device, parameters, new_state_dict)

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

    output_tensor = output_tensor.reshape(1, 120, 160, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    print("comp decoder3_2", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.36713229144766146
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
