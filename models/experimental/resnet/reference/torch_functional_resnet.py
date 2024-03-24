import torch


def ResNet50_BatchNorm2d1(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 112, 112), dtypes: torch.bfloat16; duration: 7.4 ms


def ResNet50_MaxPool2d1(input, *, parameters):
    return torch.nn.functional.max_pool2d(
        input, 3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=False
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 18.4 ms


def ResNet50_Sequential1_Bottleneck1_Conv2d2(input, *, parameters):
    weight = parameters.weight  # shapes: (64, 64, 1, 1), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 33.4 ms


def ResNet50_Sequential1_Bottleneck1_Sequential2_Conv2d3(input, *, parameters):
    weight = parameters.weight  # shapes: (256, 64, 1, 1), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 134.6 ms


def ResNet50_Sequential1_Bottleneck1_Sequential2_BatchNorm2d2(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 7.3 ms


def ResNet50_Sequential1_Bottleneck1_Sequential2(input, *, parameters):
    input = ResNet50_Sequential1_Bottleneck1_Sequential2_Conv2d3(
        input=input, parameters=parameters[0]
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 135.6 ms
    return ResNet50_Sequential1_Bottleneck1_Sequential2_BatchNorm2d2(
        input=input, parameters=parameters[1]
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 9.2 ms


def ResNet50_Sequential1_Bottleneck1_BatchNorm2d3(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 655.4 µs


def ResNet50_Sequential1_Bottleneck1_Conv2d4(input, *, parameters):
    weight = parameters.weight  # shapes: (64, 64, 3, 3), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 392.7 ms


def ResNet50_Sequential1_Bottleneck1_BatchNorm2d4(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 8.5 ms


def ResNet50_Sequential1_Bottleneck1_Conv2d5(input, *, parameters):
    weight = parameters.weight  # shapes: (256, 64, 1, 1), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 135.0 ms


def ResNet50_Sequential1_Bottleneck1_BatchNorm2d5(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 8.3 ms


def ResNet50_Sequential1_Bottleneck1(x, *, parameters):
    variable_0 = ResNet50_Sequential1_Bottleneck1_Conv2d2(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 34.2 ms
    variable_1 = ResNet50_Sequential1_Bottleneck1_Sequential2(
        input=x, parameters=parameters.downsample
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 148.3 ms
    variable_0 = ResNet50_Sequential1_Bottleneck1_BatchNorm2d3(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 2.6 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 534.1 µs
    variable_0 = ResNet50_Sequential1_Bottleneck1_Conv2d4(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 393.5 ms
    variable_0 = ResNet50_Sequential1_Bottleneck1_BatchNorm2d4(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 10.5 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 906.5 µs
    variable_0 = ResNet50_Sequential1_Bottleneck1_Conv2d5(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 136.0 ms
    variable_0 = ResNet50_Sequential1_Bottleneck1_BatchNorm2d5(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 10.2 ms
    variable_2 = torch.Tensor.add_(
        variable_0, variable_1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 807.5 µs
    variable_3 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 903.4 µs
    return variable_3


def ResNet50_Sequential1_Bottleneck2_Conv2d6(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (64, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 168.6 ms
    return variable_1


def ResNet50_Sequential1_Bottleneck2_BatchNorm2d6(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 4.5 ms


def ResNet50_Sequential1_Bottleneck2_Conv2d7(input, *, parameters):
    weight = parameters.weight  # shapes: (64, 64, 3, 3), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 393.6 ms


def ResNet50_Sequential1_Bottleneck2_BatchNorm2d7(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 12.7 ms


def ResNet50_Sequential1_Bottleneck2_Conv2d8(input, *, parameters):
    return torch.conv2d(
        input, parameters.weight, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 137.0 ms


def ResNet50_Sequential1_Bottleneck2_BatchNorm2d8(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 7.2 ms


def ResNet50_Sequential1_Bottleneck2(config, x, *, parameters):
    variable_0 = ResNet50_Sequential1_Bottleneck2_Conv2d6(
        config, input=x, parameters=parameters.conv1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 164.8 ms
    variable_0 = ResNet50_Sequential1_Bottleneck2_BatchNorm2d6(
        config, input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 5.1 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 657.3 µs
    variable_0 = ResNet50_Sequential1_Bottleneck2_Conv2d7(
        config, input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 386.4 ms
    variable_0 = ResNet50_Sequential1_Bottleneck2_BatchNorm2d7(
        config, input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 5.6 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 691.4 µs
    variable_0 = ResNet50_Sequential1_Bottleneck2_Conv2d8(
        config, input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 134.7 ms
    variable_0 = ResNet50_Sequential1_Bottleneck2_BatchNorm2d8(
        config, input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 9.8 ms
    variable_1 = torch.Tensor.add_(variable_0, x)  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 1.9 ms
    variable_2 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 942.5 µs
    return variable_2


def ResNet50_Sequential1_Bottleneck3_Conv2d9(input, *, parameters):
    weight = parameters.weight  # shapes: (64, 256, 1, 1), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 168.4 ms


def ResNet50_Sequential1_Bottleneck3_BatchNorm2d9(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 1.8 ms


def ResNet50_Sequential1_Bottleneck3_Conv2d10(input, *, parameters):
    weight = parameters.weight  # shapes: (64, 64, 3, 3), dtypes: torch.bfloat16
    return torch.conv2d(
        input, weight, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 394.3 ms


def ResNet50_Sequential1_Bottleneck3_BatchNorm2d10(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 2.5 ms


def ResNet50_Sequential1_Bottleneck3_Conv2d11(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 64, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 135.0 ms
    return variable_1


def ResNet50_Sequential1_Bottleneck3_BatchNorm2d11(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 7.2 ms


def ResNet50_Sequential1_Bottleneck3(x, *, parameters):
    variable_0 = ResNet50_Sequential1_Bottleneck3_Conv2d9(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 169.2 ms
    variable_0 = ResNet50_Sequential1_Bottleneck3_BatchNorm2d9(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 4.4 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 588.4 µs
    variable_0 = ResNet50_Sequential1_Bottleneck3_Conv2d10(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 395.3 ms
    variable_0 = ResNet50_Sequential1_Bottleneck3_BatchNorm2d10(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 4.8 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 686.6 µs
    variable_0 = ResNet50_Sequential1_Bottleneck3_Conv2d11(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 136.0 ms
    variable_0 = ResNet50_Sequential1_Bottleneck3_BatchNorm2d11(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 9.1 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 940.1 µs
    variable_2 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 896.5 µs
    return variable_2


def ResNet50_Sequential1(input, *, parameters):
    input = ResNet50_Sequential1_Bottleneck1(
        x=input, parameters=parameters[0]
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 754.6 ms
    input = ResNet50_Sequential1_Bottleneck2(
        x=input, parameters=parameters[1]
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 752.6 ms
    return ResNet50_Sequential1_Bottleneck3(
        x=input, parameters=parameters[2]
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 736.7 ms


def ResNet50_Sequential3_Bottleneck4_Sequential4_Conv2d12(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 264.6 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck4_Sequential4_BatchNorm2d12(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 4.5 ms
    return variable_4


def ResNet50_Sequential3_Bottleneck4_Sequential4(input, *, parameters):
    input = ResNet50_Sequential3_Bottleneck4_Sequential4_Conv2d12(
        input=input, parameters=parameters[0]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 265.4 ms
    variable_0 = ResNet50_Sequential3_Bottleneck4_Sequential4_BatchNorm2d12(
        input=input, parameters=parameters[1]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 6.5 ms
    return variable_0


def ResNet50_Sequential3_Bottleneck4_Conv2d13(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 128, 56, 56), dtypes: torch.bfloat16; duration: 330.1 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck4_BatchNorm2d13(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 56, 56), dtypes: torch.bfloat16; duration: 15.8 ms


def ResNet50_Sequential3_Bottleneck4_Conv2d14(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 128, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [1, 1], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 384.3 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck4_BatchNorm2d14(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 2.0 ms


def ResNet50_Sequential3_Bottleneck4_Conv2d15(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 128, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 124.2 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck4_BatchNorm2d15(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 4.7 ms


def ResNet50_Sequential3_Bottleneck4(x, *, parameters):
    variable_0 = ResNet50_Sequential3_Bottleneck4_Sequential4(
        input=x, parameters=parameters.downsample
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 275.5 ms
    variable_1 = ResNet50_Sequential3_Bottleneck4_Conv2d13(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 128, 56, 56), dtypes: torch.bfloat16; duration: 331.0 ms
    variable_1 = ResNet50_Sequential3_Bottleneck4_BatchNorm2d13(
        input=variable_1, parameters=parameters.bn1
    )  # shapes: (20, 128, 56, 56), dtypes: torch.bfloat16; duration: 17.7 ms
    variable_1 = torch.nn.functional.relu(
        input=variable_1, inplace=True
    )  # shapes: (20, 128, 56, 56), dtypes: torch.bfloat16; duration: 8.1 ms
    variable_1 = ResNet50_Sequential3_Bottleneck4_Conv2d14(
        input=variable_1, parameters=parameters.conv2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 385.1 ms
    variable_1 = ResNet50_Sequential3_Bottleneck4_BatchNorm2d14(
        input=variable_1, parameters=parameters.bn2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 4.2 ms
    variable_1 = torch.nn.functional.relu(
        input=variable_1, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 549.6 µs
    variable_1 = ResNet50_Sequential3_Bottleneck4_Conv2d15(
        input=variable_1, parameters=parameters.conv3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 125.0 ms
    variable_1 = ResNet50_Sequential3_Bottleneck4_BatchNorm2d15(
        input=variable_1, parameters=parameters.bn3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 6.6 ms
    variable_2 = torch.Tensor.add_(
        variable_1, variable_0
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 485.2 µs
    variable_3 = torch.nn.functional.relu(
        input=variable_2, inplace=True
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 657.3 µs
    return variable_3


def ResNet50_Sequential3_Bottleneck5_Conv2d16(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 162.3 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck5_BatchNorm2d16(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.4 ms


def ResNet50_Sequential3_Bottleneck5_Conv2d17(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 128, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 374.2 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck5_BatchNorm2d17(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.4 ms


def ResNet50_Sequential3_Bottleneck5_Conv2d18(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 128, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 124.3 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck5_BatchNorm2d18(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 4.5 ms


def ResNet50_Sequential3_Bottleneck5(x, *, parameters):
    variable_0 = ResNet50_Sequential3_Bottleneck5_Conv2d16(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 163.1 ms
    variable_0 = ResNet50_Sequential3_Bottleneck5_BatchNorm2d16(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.6 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 496.1 µs
    variable_0 = ResNet50_Sequential3_Bottleneck5_Conv2d17(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 375.1 ms
    variable_0 = ResNet50_Sequential3_Bottleneck5_BatchNorm2d17(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.4 ms
    variable_0 = torch.nn.functional.relu(
        input=variable_0, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 516.4 µs
    variable_0 = ResNet50_Sequential3_Bottleneck5_Conv2d18(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 125.1 ms
    variable_0 = ResNet50_Sequential3_Bottleneck5_BatchNorm2d18(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 6.5 ms
    variable_1 = torch.Tensor.add_(variable_0, x)  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 1.5 ms
    variable_2 = torch.nn.functional.relu(
        input=variable_1, inplace=True
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 645.6 µs
    return variable_2


def ResNet50_Sequential3_Bottleneck6_Conv2d19(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 151.6 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck6_BatchNorm2d19(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.1 ms


def ResNet50_Sequential3_Bottleneck6_ReLU17(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 108.5 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck6_Conv2d20(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 128, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 374.7 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck6_BatchNorm2d20(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.1 ms


def ResNet50_Sequential3_Bottleneck6_ReLU18(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 106.3 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck6_Conv2d21(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 128, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 124.0 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck6_BatchNorm2d21(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 4.6 ms


def ResNet50_Sequential3_Bottleneck6_ReLU19(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 272.8 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck6(x, *, parameters):
    variable_0 = ResNet50_Sequential3_Bottleneck6_Conv2d19(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 153.5 ms
    variable_0 = ResNet50_Sequential3_Bottleneck6_BatchNorm2d19(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.2 ms
    variable_0 = ResNet50_Sequential3_Bottleneck6_ReLU17(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 486.1 µs
    variable_0 = ResNet50_Sequential3_Bottleneck6_Conv2d20(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 375.7 ms
    variable_0 = ResNet50_Sequential3_Bottleneck6_BatchNorm2d20(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.4 ms
    variable_0 = ResNet50_Sequential3_Bottleneck6_ReLU18(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 488.8 µs
    variable_0 = ResNet50_Sequential3_Bottleneck6_Conv2d21(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 124.8 ms
    variable_0 = ResNet50_Sequential3_Bottleneck6_BatchNorm2d21(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 6.6 ms
    variable_1 = torch.Tensor.add_(variable_0, x)  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 1.4 ms
    variable_2 = ResNet50_Sequential3_Bottleneck6_ReLU19(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 646.6 µs
    return variable_2


def ResNet50_Sequential3_Bottleneck7_Conv2d22(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 150.1 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck7_BatchNorm2d22(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.1 ms
    return variable_4


def ResNet50_Sequential3_Bottleneck7_ReLU20(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 106.6 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck7_Conv2d23(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (128, 128, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 372.5 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck7_BatchNorm2d23(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 1.2 ms
    return variable_4


def ResNet50_Sequential3_Bottleneck7_ReLU21(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 104.2 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck7_Conv2d24(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 128, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 124.2 ms
    return variable_1


def ResNet50_Sequential3_Bottleneck7_BatchNorm2d24(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 4.3 ms


def ResNet50_Sequential3_Bottleneck7_ReLU22(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 270.8 µs
    return variable_0


def ResNet50_Sequential3_Bottleneck7(x, *, parameters):
    variable_0 = ResNet50_Sequential3_Bottleneck7_Conv2d22(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 150.9 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7_BatchNorm2d22(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.2 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7_ReLU20(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 555.8 µs
    variable_0 = ResNet50_Sequential3_Bottleneck7_Conv2d23(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 373.4 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7_BatchNorm2d23(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 3.3 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7_ReLU21(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 128, 28, 28), dtypes: torch.bfloat16; duration: 486.4 µs
    variable_0 = ResNet50_Sequential3_Bottleneck7_Conv2d24(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 125.1 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7_BatchNorm2d24(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 6.2 ms
    variable_1 = torch.Tensor.add_(variable_0, x)  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 1.4 ms
    variable_2 = ResNet50_Sequential3_Bottleneck7_ReLU22(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 644.4 µs
    return variable_2


def ResNet50_Sequential3(input, *, parameters):
    input = ResNet50_Sequential3_Bottleneck4(
        x=input, parameters=parameters[0]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 1.2 s
    input = ResNet50_Sequential3_Bottleneck5(
        x=input, parameters=parameters[1]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 695.1 ms
    input = ResNet50_Sequential3_Bottleneck6(
        x=input, parameters=parameters[2]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 685.3 ms
    variable_0 = ResNet50_Sequential3_Bottleneck7(
        x=input, parameters=parameters[3]
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 680.2 ms
    return variable_0


def ResNet50_Sequential5_Bottleneck8_Conv2d25(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 325.8 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck8_Sequential6_Conv2d26(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 348.5 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck8_Sequential6_BatchNorm2d25(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 13.5 ms
    return variable_4


def ResNet50_Sequential5_Bottleneck8_Sequential6(input, *, parameters):
    input = ResNet50_Sequential5_Bottleneck8_Sequential6_Conv2d26(
        input=input, parameters=parameters[0]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 349.4 ms
    return ResNet50_Sequential5_Bottleneck8_Sequential6_BatchNorm2d25(
        input=input, parameters=parameters[1]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 15.5 ms


def ResNet50_Sequential5_Bottleneck8_BatchNorm2d26(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 2.2 ms


def ResNet50_Sequential5_Bottleneck8_ReLU23(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 158.5 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck8_Conv2d27(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 404.5 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck8_BatchNorm2d27(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 1.2 ms


def ResNet50_Sequential5_Bottleneck8_ReLU24(input, *, parameters):
    return torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 79.2 µs


def ResNet50_Sequential5_Bottleneck8_Conv2d28(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 130.9 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck8_BatchNorm2d28(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.1 ms
    return variable_4


def ResNet50_Sequential5_Bottleneck8_ReLU25(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 230.6 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck8(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck8_Conv2d25(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 326.7 ms
    variable_1 = ResNet50_Sequential5_Bottleneck8_Sequential6(
        input=x, parameters=parameters.downsample
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 368.4 ms
    variable_0 = ResNet50_Sequential5_Bottleneck8_BatchNorm2d26(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 4.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck8_ReLU23(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 28, 28), dtypes: torch.bfloat16; duration: 548.1 µs
    variable_0 = ResNet50_Sequential5_Bottleneck8_Conv2d27(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 405.3 ms
    variable_0 = ResNet50_Sequential5_Bottleneck8_BatchNorm2d27(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 3.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck8_ReLU24(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 454.4 µs
    variable_0 = ResNet50_Sequential5_Bottleneck8_Conv2d28(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 131.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck8_BatchNorm2d28(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 5.1 ms
    variable_2 = torch.Tensor.add_(
        variable_0, variable_1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 698.8 µs
    variable_3 = ResNet50_Sequential5_Bottleneck8_ReLU25(
        input=variable_2, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 611.5 µs
    return variable_3


def ResNet50_Sequential5_Bottleneck9_Conv2d29(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 131.0 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck9_BatchNorm2d29(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 867.4 µs


def ResNet50_Sequential5_Bottleneck9_ReLU26(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 83.2 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck9_Conv2d30(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 404.7 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck9_BatchNorm2d30(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 860.0 µs


def ResNet50_Sequential5_Bottleneck9_ReLU27(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 75.8 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck9_Conv2d31(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 132.4 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck9_BatchNorm2d31(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.2 ms


def ResNet50_Sequential5_Bottleneck9_ReLU28(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 180.5 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck9(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck9_Conv2d29(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 131.9 ms
    variable_0 = ResNet50_Sequential5_Bottleneck9_BatchNorm2d29(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck9_ReLU26(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 450.4 µs
    variable_0 = ResNet50_Sequential5_Bottleneck9_Conv2d30(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 405.6 ms
    variable_0 = ResNet50_Sequential5_Bottleneck9_BatchNorm2d30(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck9_ReLU27(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 448.7 µs
    variable_0 = ResNet50_Sequential5_Bottleneck9_Conv2d31(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 133.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck9_BatchNorm2d31(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 5.2 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 942.2 µs
    variable_2 = ResNet50_Sequential5_Bottleneck9_ReLU28(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 551.7 µs
    return variable_2


def ResNet50_Sequential5_Bottleneck10_Conv2d32(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 142.3 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck10_BatchNorm2d32(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 1.3 ms


def ResNet50_Sequential5_Bottleneck10_ReLU29(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 83.2 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck10_Conv2d33(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 402.2 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck10_BatchNorm2d33(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 976.1 µs


def ResNet50_Sequential5_Bottleneck10_ReLU30(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 78.7 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck10_Conv2d34(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 131.2 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck10_BatchNorm2d34(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 1.7 ms


def ResNet50_Sequential5_Bottleneck10_ReLU31(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 159.7 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck10(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck10_Conv2d32(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 143.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck10_BatchNorm2d32(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 3.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck10_ReLU29(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 455.4 µs
    variable_0 = ResNet50_Sequential5_Bottleneck10_Conv2d33(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 403.0 ms
    variable_0 = ResNet50_Sequential5_Bottleneck10_BatchNorm2d33(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.9 ms
    variable_0 = ResNet50_Sequential5_Bottleneck10_ReLU30(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 453.9 µs
    variable_0 = ResNet50_Sequential5_Bottleneck10_Conv2d34(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 132.0 ms
    variable_0 = ResNet50_Sequential5_Bottleneck10_BatchNorm2d34(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.7 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 908.1 µs
    variable_2 = ResNet50_Sequential5_Bottleneck10_ReLU31(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 523.8 µs
    return variable_2


def ResNet50_Sequential5_Bottleneck11_Conv2d35(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 147.4 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck11_BatchNorm2d35(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 954.2 µs
    return variable_4


def ResNet50_Sequential5_Bottleneck11_ReLU32(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 92.0 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck11_Conv2d36(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 404.1 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck11_BatchNorm2d36(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 687.1 µs


def ResNet50_Sequential5_Bottleneck11_ReLU33(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 94.9 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck11_Conv2d37(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 129.5 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck11_BatchNorm2d37(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 1.7 ms


def ResNet50_Sequential5_Bottleneck11_ReLU34(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 156.6 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck11(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck11_Conv2d35(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 148.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck11_BatchNorm2d35(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck11_ReLU32(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 540.7 µs
    variable_0 = ResNet50_Sequential5_Bottleneck11_Conv2d36(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 405.2 ms
    variable_0 = ResNet50_Sequential5_Bottleneck11_BatchNorm2d36(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.7 ms
    variable_0 = ResNet50_Sequential5_Bottleneck11_ReLU33(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 466.1 µs
    variable_0 = ResNet50_Sequential5_Bottleneck11_Conv2d37(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 130.4 ms
    variable_0 = ResNet50_Sequential5_Bottleneck11_BatchNorm2d37(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.8 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 835.4 µs
    variable_2 = ResNet50_Sequential5_Bottleneck11_ReLU34(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 547.2 µs
    return variable_2


def ResNet50_Sequential5_Bottleneck12_Conv2d38(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 131.0 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck12_BatchNorm2d38(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 780.8 µs


def ResNet50_Sequential5_Bottleneck12_ReLU35(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 90.6 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck12_Conv2d39(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 403.6 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck12_BatchNorm2d39(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 891.0 µs


def ResNet50_Sequential5_Bottleneck12_ReLU36(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 83.0 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck12_Conv2d40(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 131.8 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck12_BatchNorm2d40(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 1.7 ms
    return variable_4


def ResNet50_Sequential5_Bottleneck12_ReLU37(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 168.6 µs


def ResNet50_Sequential5_Bottleneck12(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck12_Conv2d38(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 131.9 ms
    variable_0 = ResNet50_Sequential5_Bottleneck12_BatchNorm2d38(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck12_ReLU35(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 459.7 µs
    variable_0 = ResNet50_Sequential5_Bottleneck12_Conv2d39(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 404.6 ms
    variable_0 = ResNet50_Sequential5_Bottleneck12_BatchNorm2d39(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck12_ReLU36(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 453.5 µs
    variable_0 = ResNet50_Sequential5_Bottleneck12_Conv2d40(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 132.6 ms
    variable_0 = ResNet50_Sequential5_Bottleneck12_BatchNorm2d40(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.7 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 735.8 µs
    variable_2 = ResNet50_Sequential5_Bottleneck12_ReLU37(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 622.3 µs
    return variable_2


def ResNet50_Sequential5_Bottleneck13_Conv2d41(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 139.0 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck13_BatchNorm2d41(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 885.2 µs


def ResNet50_Sequential5_Bottleneck13_ReLU38(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 78.4 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck13_Conv2d42(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (256, 256, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 407.9 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck13_BatchNorm2d42(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 1.1 ms


def ResNet50_Sequential5_Bottleneck13_ReLU39(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 80.8 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck13_Conv2d43(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1024, 256, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 130.9 ms
    return variable_1


def ResNet50_Sequential5_Bottleneck13_BatchNorm2d43(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 1.7 ms


def ResNet50_Sequential5_Bottleneck13_ReLU40(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 161.4 µs
    return variable_0


def ResNet50_Sequential5_Bottleneck13(x, *, parameters):
    variable_0 = ResNet50_Sequential5_Bottleneck13_Conv2d41(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 139.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13_BatchNorm2d41(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 2.9 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13_ReLU38(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 445.1 µs
    variable_0 = ResNet50_Sequential5_Bottleneck13_Conv2d42(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 408.8 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13_BatchNorm2d42(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 3.0 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13_ReLU39(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 256, 14, 14), dtypes: torch.bfloat16; duration: 520.2 µs
    variable_0 = ResNet50_Sequential5_Bottleneck13_Conv2d43(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 131.7 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13_BatchNorm2d43(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 3.9 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 702.4 µs
    variable_2 = ResNet50_Sequential5_Bottleneck13_ReLU40(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 529.5 µs
    return variable_2


def ResNet50_Sequential5(input, *, parameters):
    input = ResNet50_Sequential5_Bottleneck8(
        x=input, parameters=parameters[0]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 1.3 s
    input = ResNet50_Sequential5_Bottleneck9(
        x=input, parameters=parameters[1]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 699.5 ms
    input = ResNet50_Sequential5_Bottleneck10(
        x=input, parameters=parameters[2]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 704.8 ms
    input = ResNet50_Sequential5_Bottleneck11(
        x=input, parameters=parameters[3]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 710.1 ms
    input = ResNet50_Sequential5_Bottleneck12(
        x=input, parameters=parameters[4]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 695.4 ms
    variable_0 = ResNet50_Sequential5_Bottleneck13(
        x=input, parameters=parameters[5]
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 706.9 ms
    return variable_0


def ResNet50_Sequential7_Bottleneck14_Sequential8_Conv2d44(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (2048, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [0, 0], [1, 1], 1
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 386.3 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck14_Sequential8_BatchNorm2d44(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.2 ms


def ResNet50_Sequential7_Bottleneck14_Sequential8(input, *, parameters):
    input = ResNet50_Sequential7_Bottleneck14_Sequential8_Conv2d44(
        input=input, parameters=parameters[0]
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 387.3 ms
    variable_0 = ResNet50_Sequential7_Bottleneck14_Sequential8_BatchNorm2d44(
        input=input, parameters=parameters[1]
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 3.1 ms
    return variable_0


def ResNet50_Sequential7_Bottleneck14_Conv2d45(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 1024, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 329.0 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck14_BatchNorm2d45(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 1.0 ms


def ResNet50_Sequential7_Bottleneck14_ReLU41(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 100.6 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck14_Conv2d46(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 512, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [2, 2], [1, 1], [1, 1], 1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 710.3 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck14_BatchNorm2d46(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 379.1 µs


def ResNet50_Sequential7_Bottleneck14_ReLU42(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 67.9 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck14_Conv2d47(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (2048, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 183.3 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck14_BatchNorm2d47(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 10.3 ms


def ResNet50_Sequential7_Bottleneck14_ReLU43(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 112.5 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck14(x, *, parameters):
    variable_0 = ResNet50_Sequential7_Bottleneck14_Sequential8(
        input=x, parameters=parameters.downsample
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 394.3 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_Conv2d45(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 329.8 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_BatchNorm2d45(
        input=variable_1, parameters=parameters.bn1
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 3.2 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_ReLU41(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 512, 14, 14), dtypes: torch.bfloat16; duration: 471.1 µs
    variable_1 = ResNet50_Sequential7_Bottleneck14_Conv2d46(
        input=variable_1, parameters=parameters.conv2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 711.4 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_BatchNorm2d46(
        input=variable_1, parameters=parameters.bn2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 2.3 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_ReLU42(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 441.1 µs
    variable_1 = ResNet50_Sequential7_Bottleneck14_Conv2d47(
        input=variable_1, parameters=parameters.conv3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 184.1 ms
    variable_1 = ResNet50_Sequential7_Bottleneck14_BatchNorm2d47(
        input=variable_1, parameters=parameters.bn3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 12.3 ms
    variable_2 = torch.Tensor.add_(
        variable_1, variable_0
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 383.6 µs
    variable_3 = ResNet50_Sequential7_Bottleneck14_ReLU43(
        input=variable_2, parameters=parameters.relu
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 495.2 µs
    return variable_3


def ResNet50_Sequential7_Bottleneck15_Conv2d48(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 2048, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 190.3 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck15_BatchNorm2d48(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 11.6 ms


def ResNet50_Sequential7_Bottleneck15_ReLU44(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 89.4 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck15_Conv2d49(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 512, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 678.2 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck15_BatchNorm2d49(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 7.5 ms


def ResNet50_Sequential7_Bottleneck15_ReLU45(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 73.7 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck15_Conv2d50(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (2048, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 193.7 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck15_BatchNorm2d50(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.5 ms


def ResNet50_Sequential7_Bottleneck15_ReLU46(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 103.7 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck15(x, *, parameters):
    variable_0 = ResNet50_Sequential7_Bottleneck15_Conv2d48(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 191.1 ms
    variable_0 = ResNet50_Sequential7_Bottleneck15_BatchNorm2d48(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 13.4 ms
    variable_0 = ResNet50_Sequential7_Bottleneck15_ReLU44(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 548.1 µs
    variable_0 = ResNet50_Sequential7_Bottleneck15_Conv2d49(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 679.1 ms
    variable_0 = ResNet50_Sequential7_Bottleneck15_BatchNorm2d49(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 9.5 ms
    variable_0 = ResNet50_Sequential7_Bottleneck15_ReLU45(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 462.1 µs
    variable_0 = ResNet50_Sequential7_Bottleneck15_Conv2d50(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 194.7 ms
    variable_0 = ResNet50_Sequential7_Bottleneck15_BatchNorm2d50(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 3.6 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 500.7 µs
    variable_2 = ResNet50_Sequential7_Bottleneck15_ReLU46(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 472.1 µs
    return variable_2


def ResNet50_Sequential7_Bottleneck16_Conv2d51(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 2048, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 200.7 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck16_BatchNorm2d51(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 544.3 µs


def ResNet50_Sequential7_Bottleneck16_ReLU47(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 67.5 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck16_Conv2d52(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (512, 512, 3, 3), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [1, 1], [1, 1], 1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 681.3 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck16_BatchNorm2d52(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 510.2 µs


def ResNet50_Sequential7_Bottleneck16_ReLU48(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 71.3 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck16_Conv2d53(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (2048, 512, 1, 1), dtypes: torch.bfloat16
    variable_1 = torch.conv2d(
        input, variable_0, None, [1, 1], [0, 0], [1, 1], 1
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 194.4 ms
    return variable_1


def ResNet50_Sequential7_Bottleneck16_BatchNorm2d53(input, *, parameters):
    return torch.nn.functional.batch_norm(
        input=input,
        running_mean=parameters.running_mean,
        running_var=parameters.running_var,
        weight=parameters.weight,
        bias=parameters.bias,
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.4 ms


def ResNet50_Sequential7_Bottleneck16_ReLU49(input, *, parameters):
    variable_0 = torch.nn.functional.relu(
        input=input, inplace=True
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 99.7 µs
    return variable_0


def ResNet50_Sequential7_Bottleneck16(x, *, parameters):
    variable_0 = ResNet50_Sequential7_Bottleneck16_Conv2d51(
        input=x, parameters=parameters.conv1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 201.5 ms
    variable_0 = ResNet50_Sequential7_Bottleneck16_BatchNorm2d51(
        input=variable_0, parameters=parameters.bn1
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 2.5 ms
    variable_0 = ResNet50_Sequential7_Bottleneck16_ReLU47(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 446.6 µs
    variable_0 = ResNet50_Sequential7_Bottleneck16_Conv2d52(
        input=variable_0, parameters=parameters.conv2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 682.3 ms
    variable_0 = ResNet50_Sequential7_Bottleneck16_BatchNorm2d52(
        input=variable_0, parameters=parameters.bn2
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 2.4 ms
    variable_0 = ResNet50_Sequential7_Bottleneck16_ReLU48(
        input=variable_0, parameters=parameters.relu
    )  # shapes: (20, 512, 7, 7), dtypes: torch.bfloat16; duration: 461.1 µs
    variable_0 = ResNet50_Sequential7_Bottleneck16_Conv2d53(
        input=variable_0, parameters=parameters.conv3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 195.3 ms
    variable_0 = ResNet50_Sequential7_Bottleneck16_BatchNorm2d53(
        input=variable_0, parameters=parameters.bn3
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 3.3 ms
    variable_1 = torch.Tensor.add_(
        variable_0, x
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 558.4 µs
    variable_2 = ResNet50_Sequential7_Bottleneck16_ReLU49(
        input=variable_1, parameters=parameters.relu
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 465.4 µs
    return variable_2


def ResNet50_Sequential7(input, *, parameters):
    input = ResNet50_Sequential7_Bottleneck14(
        x=input, parameters=parameters[0]
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.7 s
    input = ResNet50_Sequential7_Bottleneck15(
        x=input, parameters=parameters[1]
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.1 s
    variable_0 = ResNet50_Sequential7_Bottleneck16(
        x=input, parameters=parameters[2]
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 1.1 s
    return variable_0


def ResNet50_AdaptiveAvgPool2d1(input, *, parameters):
    variable_0 = torch.nn.functional.adaptive_avg_pool2d(
        input=input, output_size=[1, 1]
    )  # shapes: (20, 2048, 1, 1), dtypes: torch.bfloat16; duration: 5.6 ms
    return variable_0


def ResNet50_Linear1(input, *, parameters):
    variable_0 = parameters.weight  # shapes: (1000, 2048), dtypes: torch.bfloat16
    variable_1 = parameters.bias  # shapes: (1000,), dtypes: torch.bfloat16
    variable_2 = torch.nn.functional.linear(
        input, variable_0, variable_1
    )  # shapes: (20, 1000), dtypes: torch.bfloat16; duration: 26.9 ms
    return variable_2


# Input x expected to have the shape (20, 3, 224, 224)
# Output will have the shape (20, 1000)
def ResNet50(x, *, parameters):
    x = torch.conv2d(
        x, parameters.conv1.weight, None, [2, 2], [3, 3], [1, 1], 1
    )  # shapes: (20, 64, 112, 112), dtypes: torch.bfloat16; duration: 604.5 ms
    x = torch.nn.functional.batch_norm(
        input=x,
        running_mean=parameters.bn1.running_mean,
        running_var=parameters.bn1.running_var,
        weight=parameters.bn1.weight,
        bias=parameters.bn1.bias,
    )  # shapes: (20, 64, 112, 112), dtypes: torch.bfloat16; duration: 9.4 ms
    x = torch.nn.functional.relu(
        input=x, inplace=True
    )  # shapes: (20, 64, 112, 112), dtypes: torch.bfloat16; duration: 1.2 ms
    x = torch.nn.functional.max_pool2d(
        input=x, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=False
    )  # shapes: (20, 64, 56, 56), dtypes: torch.bfloat16; duration: 18.9 ms
    x = ResNet50_Sequential1(
        input=x, parameters=parameters.layer1
    )  # shapes: (20, 256, 56, 56), dtypes: torch.bfloat16; duration: 2.3 s
    x = ResNet50_Sequential3(
        input=x, parameters=parameters.layer2
    )  # shapes: (20, 512, 28, 28), dtypes: torch.bfloat16; duration: 3.2 s
    x = ResNet50_Sequential5(
        input=x, parameters=parameters.layer3
    )  # shapes: (20, 1024, 14, 14), dtypes: torch.bfloat16; duration: 4.8 s
    x = ResNet50_Sequential7(
        input=x, parameters=parameters.layer4
    )  # shapes: (20, 2048, 7, 7), dtypes: torch.bfloat16; duration: 3.9 s
    x = ResNet50_AdaptiveAvgPool2d1(
        input=x, parameters=parameters.avgpool
    )  # shapes: (20, 2048, 1, 1), dtypes: torch.bfloat16; duration: 6.0 ms
    x = torch.flatten(x, 1)  # shapes: (20, 2048), dtypes: torch.bfloat16; duration: 24.6 µs
    return ResNet50_Linear1(
        input=x, parameters=parameters.fc
    )  # shapes: (20, 1000), dtypes: torch.bfloat16; duration: 28.0 ms


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, torch.nn.modules.batchnorm.BatchNorm2d):
        parameters["running_mean"] = torch_model.running_mean
        parameters["running_var"] = torch_model.running_var
        parameters["weight"] = torch_model.weight.data
        parameters["bias"] = torch_model.bias.data
        # Todo: fold parameters for running_var and running_mean into parameters needed for conv here
        return parameters

    if isinstance(torch_model, torch.nn.modules.Sequential):
        return parameters
    return parameters
