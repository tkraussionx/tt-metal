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
        if isinstance(model, UNet):
            ttnn_module_args["encoder3_c1"] = ttnn_module_args.encoder3["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder3[0], model.encoder3[1])

            update_ttnn_module_args(ttnn_module_args["encoder3_c1"])
            ttnn_module_args["encoder3_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder3_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder3_c1"]["use_shallow_conv_variant"] = False

            parameters["encoder3_c1"], encoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder3_c1"], return_parallel_config=True
            )

            return parameters

    return custom_preprocessor


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.encoder3(x)
        return output


class TtUnet:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc3_1 = parameters.encoder3_c1

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc3_1.conv.input_sharded_memory_config)
        output_tensor_enc3_1 = self.enc3_1(input_tensor)

        return output_tensor_enc3_1


@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
@skip_for_wormhole_b0()
def test_unet(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("encoder3."))}

    torch_model = UNet()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 64, 120, 160)  # (1, 3, 160, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet(device, parameters, new_state_dict)

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

    print("comp,", comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=0.99))  # 0.9992638200717393
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
