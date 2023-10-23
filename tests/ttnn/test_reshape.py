import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_reshape(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.reshape(1, 1, k, m)
    tt_output = ttnn.reshape(activations, (1, 1, k, m))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("k", [2 * 32])
def test_reshape_negative_1(device, m, k):
    activations = ttnn.random(shape=(1, 1, m, k))
    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.reshape(-1)
    # activations.reshape(-1) is currently not supported
    tt_output = ttnn.reshape(activations, (1, 1, m, -1))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("n", [32, 32])
@pytest.mark.parametrize("c", [2 * 32, 2 * 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("w", [1, 4])
def test_reshape_in_4D(device, n, c, h, w):
    activations = ttnn.random(shape=(n, c, h, w))
    torch_activations = ttnn.to_torch(activations)
    torch_output = torch_activations.reshape(h, w, n, c)
    tt_output = ttnn.reshape(activations, (h, w, n, c))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
