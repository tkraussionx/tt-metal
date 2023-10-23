import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc, update_process_id
import tt_lib as ttl


@pytest.mark.parametrize("s", [3])
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [4 * 32])
def test_add_scalar(device, s, h, w):
    a = ttnn.random(shape=(1, 1, h, w))
    torch_a = ttnn.to_torch(a)
    expected_pytorch_result = torch_a + s
    actual_tt_result = a + s
    actual_pytorch_result = ttnn.to_torch(actual_tt_result)
    assert_with_pcc(expected_pytorch_result, actual_pytorch_result, 0.99)


@pytest.mark.parametrize("alpha", [0.42])
@pytest.mark.parametrize("scalar_input_tensor_b", [0.5])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("w", [4])
def test_add_scalar_and_alpha(device, alpha, scalar_input_tensor_b, h, w):
    update_process_id()
    a = ttnn.random(shape=(1, 1, h, w))
    torch_a = ttnn.to_torch(a)
    expected_pytorch_result = torch.add(torch_a, scalar_input_tensor_b, alpha=alpha)
    expected_tt_result = ttnn.from_torch(expected_pytorch_result, ttl.tensor.DataType.BFLOAT16).cpu()
    actual_tt_result = ttnn.add(a, scalar_input_tensor_b, alpha=alpha)
    actual_pytorch_result = ttnn.to_torch(actual_tt_result)
    print(expected_pytorch_result)
    print(actual_pytorch_result)
    assert_with_pcc(expected_pytorch_result, actual_pytorch_result, 0.99999)


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
@pytest.mark.parametrize("n", [4 * 32])
def test_add(device, m, k, n):
    a = ttnn.random(shape=(1, 1, m, k))
    b = ttnn.random(shape=(1, 1, m, k))
    torch_a = ttnn.to_torch(a)
    torch_b = ttnn.to_torch(b)
    torch_output = torch.add(torch_a, torch_b)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
@pytest.mark.parametrize("n", [4 * 32])
@pytest.mark.parametrize("l", [4 * 32])
def test_add_4D(device, m, k, n, l):
    a = ttnn.random(shape=(m, k, n, l))
    b = ttnn.random(shape=(m, k, n, l))
    torch_a = ttnn.to_torch(a)
    torch_b = ttnn.to_torch(b)
    torch_output = torch.add(torch_a, torch_b)
    tt_output = ttnn.add(a, b)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
