import torch
import torch.nn.functional as F


def downsample1(input):
    # First convolutional block
    x1 = F.conv2d(input, weight=torch.randn(32, 3, 3, 3, requires_grad=True), stride=1, padding=1, bias=None)
    x1_b = F.batch_norm(
        x1, running_mean=torch.zeros(32), running_var=torch.ones(32), weight=None, bias=None, training=True
    )
    x1_m = F.relu(x1_b, inplace=True)

    # Second convolutional block
    x2 = F.conv2d(x1_m, weight=torch.randn(64, 32, 3, 3, requires_grad=True), stride=2, padding=1, bias=None)
    x2_b = F.batch_norm(
        x2, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x2_m = F.relu(x2_b, inplace=True)

    # Third convolutional block
    x3 = F.conv2d(x2_m, weight=torch.randn(64, 64, 1, 1, requires_grad=True), stride=1, padding=0, bias=None)
    x3_b = F.batch_norm(
        x3, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x3_m = F.relu(x3_b, inplace=True)

    # Fourth convolutional block
    x4 = F.conv2d(x2_m, weight=torch.randn(64, 64, 1, 1, requires_grad=True), stride=1, padding=0, bias=None)
    x4_b = F.batch_norm(
        x4, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x4_m = F.relu(x4_b, inplace=True)

    # Fifth convolutional block
    x5 = F.conv2d(x4_m, weight=torch.randn(32, 64, 1, 1, requires_grad=True), stride=1, padding=0, bias=None)
    x5_b = F.batch_norm(
        x5, running_mean=torch.zeros(32), running_var=torch.ones(32), weight=None, bias=None, training=True
    )
    x5_m = F.relu(x5_b, inplace=True)

    # Sixth convolutional block
    x6 = F.conv2d(x5_m, weight=torch.randn(64, 32, 3, 3, requires_grad=True), stride=1, padding=1, bias=None)
    x6_b = F.batch_norm(
        x6, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x6_m = F.relu(x6_b, inplace=True)
    x6_m = x6_m + x4_m

    # Seventh convolutional block
    x7 = F.conv2d(x6_m, weight=torch.randn(64, 64, 1, 1, requires_grad=True), stride=1, padding=0, bias=None)
    x7_b = F.batch_norm(
        x7, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x7_m = F.relu(x7_b, inplace=True)
    x7_m = torch.cat([x7_m, x3_m], dim=1)

    # Eighth convolutional block
    x8 = F.conv2d(x7_m, weight=torch.randn(64, 128, 1, 1, requires_grad=True), stride=1, padding=0, bias=None)
    x8_b = F.batch_norm(
        x8, running_mean=torch.zeros(64), running_var=torch.ones(64), weight=None, bias=None, training=True
    )
    x8_m = F.relu(x8_b, inplace=True)

    return x8_m


# Example usage
input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
output_tensor = downsample1(input_tensor)
print(output_tensor.shape)
