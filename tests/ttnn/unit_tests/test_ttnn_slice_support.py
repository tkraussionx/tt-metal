import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# slice alternate elements in a given tensor
def test_slice(device):
    torch_input = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[::2]  # tensor([0, 2, 4, 6, 8])
    ttnn_output = ttnn_input[::2]
    ttnn_output = ttnn.to_torch(ttnn_output)  # torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert_with_pcc(
        torch_output, ttnn_output, 0.99
    )  # AssertionError: list(expected_pytorch_result.shape)=[5] vs list(actual_pytorch_result.shape)=[10]


def test_slice_usecase1(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, ::2]  # RuntimeError: Invalid slice type!
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase2(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., ::2, 1::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., ::2, 1::2]  # RuntimeError: Invalid slice type!
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase3(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., 1::2, ::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., 1::2, ::2]  # RuntimeError: Invalid slice type!
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_slice_usecase4(device):
    torch_input = torch.randn(1, 3, 640, 640)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)

    torch_output = torch_input[..., 1::2, 1::2]  # torch_output shape: [1, 3, 320, 320]
    ttnn_output = ttnn_input[..., 1::2, 1::2]  # RuntimeError: Invalid slice type!
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
