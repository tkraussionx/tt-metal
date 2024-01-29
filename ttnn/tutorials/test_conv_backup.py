import torch
import ttnn
import torch.nn.functional as F


# Conv functional in pytorch
def convolution_layer(input_tensor, weight, bias, kernel_size=3, stride=1, padding=0):
    # Check if the input tensor and weight have compatible shapes
    if input_tensor.size(1) != weight.size(1):
        raise ValueError("Input tensor and weight should have the same number of input channels")
    # Perform the convolution with bias and specified kernel size
    conv_result = F.conv2d(input_tensor, weight, bias=bias, stride=stride, padding=padding)
    return conv_result


# Example usage
batch_size = 4
# input_channels = 3
input_channels = 16
# output_channels = 16
output_channels = 32
input_height = 1056
input_width = 160
# input_height = 528
# input_width = 80
# Initialize input tensor, weight, and bias
input_tensor = torch.randn((batch_size, input_channels, input_height, input_width))
input_tensor[:, 3:, :, :] = 0
# kernel_size = 3
weight = torch.randn((output_channels, input_channels, 3, 3))
weight[:, 3:, :, :] = 0
print("the shape of the weight is: ", weight.size())
print("some random check: ", weight[1, 2, 1, 1])
# bias = torch.randn((output_channels,))
conv_bias_shape = [1, 1, 1, output_channels]
bias = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
# Specify kernel size
# Apply convolution layer with bias and kernel size
bias_torch = bias
conv_result = convolution_layer(input_tensor, weight, bias.reshape(-1), stride=1, padding=1)
# Print the result shape
print("Convolution Result Shape:", conv_result.shape)
# conv_bias_shape = [1, 1, 1, output_channels]
# torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

# input_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16)
# input_tensor = conv.copy_input_to_device(input_tensor)
import ttnn

# bias_tt = ttnn.from_torch(bias, ttnn.float32)
device_id = 0
device = ttnn.open(device_id)

# weight = torch.randn(output_channels, input_channels, 3, 3)
# bias = torch.randn((output_channels,))
# conv = ttnn.Conv2D(16, 64, (3, 3))
tt_weight_tensor = ttnn.from_torch(weight, ttnn.bfloat16)
# bias = ttnn.from_torch(bias, ttnn.bfloat16)
# tt_weight_tensor = ttnn.to_device(tt_weight_tensor, device)#conv.copy_input_to_device(input=tt_weight_tensor)
import ttnn


# input_channels=3
input_channels = 16
# output_channels = 16
output_channels = 32
kernel_size = (3, 3)
stride = (1, 1)
padding = (0, 0)
# dtype = activations_dtype = ttnn.bfloat16
dtype = activations_dtype = ttnn.bfloat8_b
device = device
use_1d_systolic_array = True
batch_size = 4
input_height = 1056
input_width = 160
reader_patterns_cache = {}
weight = tt_weight_tensor
# bias = bias_tt
math_fidelity = ttnn.MathFidelity.LoFi
# weights_dtype = ttnn.bfloat16
weights_dtype = ttnn.bfloat8_b
bias_tt = ttnn.from_torch(bias_torch, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32)
# bias_tt = ttnn.from_torch(bias_torch, ttnn.bfloat8_b)

conv = ttnn.Conv2D(
    input_channels,
    output_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dtype=activations_dtype,
    device=device,
    use_1d_systolic_array=use_1d_systolic_array,
    batch_size=batch_size,
    input_height=input_height,
    input_width=input_width,
    reader_patterns_cache=reader_patterns_cache,
    weight=tt_weight_tensor,
    bias=bias_tt,
    math_fidelity=math_fidelity,
    weights_dtype=weights_dtype,
)

print("input_tensor: ", input_tensor.size())
input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
input_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16)
input_tensor = conv.copy_input_to_device(input_tensor)
conv_result_ttnn = conv(input_tensor)
# conv_result_ttnn = ttnn.from_device(conv_result_ttnn)
# conv_result_ttnn = ttnn.to_layout(conv_result_ttnn, ttnn.ROW_MAJOR_LAYOUT)
# conv_result_ttnn = ttnn.to_torch(conv_result_ttnn)

# assert_with_pcc(conv_result, conv_result_ttnn.to(torch_output.dtype), 0.999)
ttnn.close(device)
