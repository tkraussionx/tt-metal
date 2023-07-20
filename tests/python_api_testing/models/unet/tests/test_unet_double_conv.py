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

from models.unet.tt.unet_double_conv import TtDoubleConv
from tests.python_api_testing.models.unet.reference.unet_model import UNet


def run_test_double_conv_inference(
    device,
    first_double_conv_position,
    second_double_conv_position,
    third_double_conv_position,
    pcc,
):
    # load Unet model ================================================
    reference_model = UNet(n_channels=3, n_classes=2, bilinear=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )
    reference_model.load_state_dict(checkpoint)
    state_dict = reference_model.state_dict()

    if second_double_conv_position is None:
        double_conv = reference_model.inc.double_conv
    elif (second_double_conv_position is not None) and (
        third_double_conv_position is None
    ):
        double_conv = reference_model.up1
        double_conv = getattr(double_conv, second_double_conv_position)
        double_conv = double_conv.double_conv

    logger.debug(f"CPU model: {double_conv}")

    # get TtDoubleConv module =====================================================
    base_address = "inc.double_conv"

    tt_module = TtDoubleConv(device, reference_model, base_address, state_dict)

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 64, 64)

    # Pytorch call =========================================================
    pt_out = double_conv(test_input)

    # TT call =========================================================
    # with torch.no_grad():
    #     tt_module.eval()

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
        logger.info("test DoubleConv Passed!")
    else:
        logger.warning("test DoubleConv Failed!")

    assert does_pass


# parameters: DoubleConv position in the model
_first_double_conv_position = 0
_second_double_conv_position = None
_third_double_conv_position = None


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_double_conv_inference(pcc):
    # Initialize the device
    first_double_conv_position = _first_double_conv_position
    second_double_conv_position = _second_double_conv_position
    third_double_conv_position = _third_double_conv_position

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_double_conv_inference(
        device,
        first_double_conv_position,
        second_double_conv_position,
        third_double_conv_position,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
