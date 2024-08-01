import torch
import ttnn
import pytest
from models.experimental.model_net.decoder import decoder, last_decoder


@pytest.mark.parametrize(
    ("input_a_shape", "input_b_shape", "force_crop"),
    [
        ((1, 254, 30, 64), (1, 510, 62, 64), (4, 4)),
        ((1, 506, 58, 128), (1, 1022, 126, 64), (4, 12)),
        ((1, 1010, 114, 128), (1, 2046, 254, 64), (4, 28)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_decoder_4094x510(input_a_shape, input_b_shape, force_crop, device):
    input_a = torch.randn(input_a_shape)
    tt_input_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, device=device)
    input_b = torch.randn(input_b_shape)
    tt_input_b = ttnn.from_torch(input_b, dtype=ttnn.bfloat16, device=device)
    output = decoder(tt_input_a, tt_input_b, force_crop=force_crop, device=device)
    print("output_shape", output.shape)


@pytest.mark.parametrize(
    ("input_a_shape", "force_crop"),
    [
        ((1, 2018, 226, 128), 4),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_last_decoder_4094x510(input_a_shape, force_crop, device):
    input_a = torch.randn(input_a_shape)
    tt_input_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, device=device)
    output = last_decoder(tt_input_a, force_crop=force_crop, device=device)
    print("output_shape", output.shape)
