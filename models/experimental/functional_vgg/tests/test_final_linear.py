import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import torch_random, is_wormhole_b0


@pytest.mark.parametrize("batch_sizes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("m_size", [1])
@pytest.mark.parametrize("k_size", [4096])
@pytest.mark.parametrize("n_size", [1000])
@pytest.mark.parametrize(
    "use_bias",
    [
        True,
    ],
)
def test_linear(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    use_bias,
    *,
    device,
):
    input_shape_a = (batch_sizes, 1, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    print(input_tensor_a.shape, input_tensor_b.shape, bias.shape)

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"Linear OP PCC: {pcc_msg}")

    output_tensor = input_tensor_a @ input_tensor_b
    output_tensor = output_tensor + bias
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, output_tensor, pcc=0.99)
    logger.info(f"Matmul and Add OP PCC: {pcc_msg}")
