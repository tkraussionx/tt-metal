import sys
import pytest
import torch
from loguru import logger
from torch import nn
import tt_lib

from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from tests.python_api_testing.models.utility_functions_new import (
    comp_pcc,
    comp_allclose_and_pcc,
)

from models.unet.unet_mini_graphs import TtUnetConv2D
from tests.python_api_testing.models.unet.reference.unet_model import UNet


def run_test_conv2d_inference(device, conv2d_position, pcc):
    # load Unet model ================================================
    reference_model = UNet(n_channels=3, n_classes=2, bilinear=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )
    reference_model.load_state_dict(checkpoint)

    cpu_conv2d = reference_model.inc.double_conv[0]
    logger.debug(f"CPU model: {cpu_conv2d}")

    # get TtConv2d module =====================================================
    base_address = "inc.double_conv.0"
    state_dict = reference_model.state_dict()
    torch_model = reference_model.inc.double_conv[0]

    in_channels = torch_model.in_channels
    out_channels = torch_model.out_channels
    kernel_size = torch_model.kernel_size[0]
    stride = torch_model.stride[0]
    padding = torch_model.padding[0]
    groups = torch_model.groups
    dilation = torch_model.dilation[0]

    logger.debug(f"in_channels {in_channels}")
    logger.debug(f"out_channels {out_channels}")
    logger.debug(f"kernel_size {kernel_size}")
    logger.debug(f"stride {stride}")
    logger.debug(f"padding {padding}")
    logger.debug(f"groups {groups}")
    logger.debug(f"dilation {dilation}")

    tt_module = TtUnetConv2D(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation,
    )

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 64, 64)

    # Pytorch call =========================================================
    pt_out = cpu_conv2d(test_input)

    # TT call =========================================================
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test Conv2D Passed!")
    else:
        logger.warning("test Conv2D Failed!")

    assert does_pass


# parameters: Fire position in the model
_conv2d_position = 4


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_conv2d_inference(pcc):
    # Initialize the device
    conv2d_position = _conv2d_position

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_conv2d_inference(device, conv2d_position, pcc)
    tt_lib.device.CloseDevice(device)
