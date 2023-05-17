import torch
import libs.tt_lib as ttl
from tests.python_api_testing.models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from libs.tt_lib.fallback_ops import fallback_ops
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape, output_shape, on_device",
    (
        (torch.Size([1, 3, 6, 4]), torch.Size([3, 1, 3, 8]), True),
        (torch.Size([1, 3, 6, 4]), torch.Size([3, 1, 3, 8]), False),
        (torch.Size([1, 2, 64, 32]), torch.Size([2, 4, 128, 4]), True),
        (torch.Size([1, 2, 64, 32]), torch.Size([2, 4, 128, 4]), False),
    ),
)
def test_reshape_fallback(input_shape, output_shape, on_device):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    pt_out = torch.reshape(x, output_shape)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = fallback_ops.reshape(t0, *output_shape)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
