import ttnn
import torch
from torch import nn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from models.utility_functions import comp_allclose_and_pcc


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters = preprocess_conv2d(model.weight, model.bias, ttnn_module_args, return_parallel_config=False)
        return parameters

    return custom_preprocessor


def test():
    device = ttnn.open_device(device_id=0)
    torch_model = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
    torch_model.eval()
    torch_input_tensor = torch.randn(1, 32, 480, 640)
    torch_output_tensor = torch_model(torch_input_tensor)
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = input_tensor.to(device, parameters.conv.input_sharded_memory_config)
    output_tensor = parameters(input_tensor)
    output_tensor = parameters.copy_output_from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    print("ttnn output shape: ", output_tensor.shape)  # 1,32,480,640
    print("torch output shape: ", torch_output_tensor.shape)  # 1,1,480,640
    assert output_tensor.shape == torch_output_tensor.shape
