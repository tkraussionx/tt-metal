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

from models.unet.tt.unet_down import TtDown
from tests.python_api_testing.models.unet.reference.unet_model import UNet


def run_test_down_inference(
    device,
    in_channels,
    out_channels,
    down_position,
    pcc,
):
    # load Unet model ================================================
    reference_model = UNet(n_channels=3, n_classes=2, bilinear=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )
    reference_model.load_state_dict(checkpoint)
    reference_model.eval()
    state_dict = reference_model.state_dict()

    # get Pytorch subgraph
    cpu_module = getattr(reference_model, down_position)
    logger.debug(f"CPU model: {cpu_module}")

    # get TtDoubleConv module =====================================================
    base_address = down_position + ".maxpool_conv.1.double_conv"

    gs_module = TtDown(device, base_address, state_dict, in_channels, out_channels)

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 512, 64, 64)

    # Pytorch call ===================================================
    pt_out = cpu_module(test_input)
    logger.debug(f"pt_out shape {pt_out.shape}")

    # TT call =========================================================
    with torch.no_grad():
        gs_module.eval()
        test_input = torch2tt_tensor(test_input, device)
        tt_out = gs_module(test_input)
        tt_out = tt2torch_tensor(tt_out)

    logger.debug(f"tt_out shape {tt_out.shape}")

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test Down Passed!")
    else:
        logger.warning("test Down Failed!")

    assert does_pass


# parameters: DoubleConv position in the model
down_number = 4
_down_position = "down" + str(down_number)

_in_channels = 64 * pow(2, down_number - 1)
_out_channels = 128 * pow(2, down_number - 1)

logger.info(_in_channels)
logger.info(_out_channels)


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_down_inference(pcc):
    # Initialize the device
    in_channels = _in_channels
    out_channels = _out_channels
    down_position = _down_position

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_down_inference(
        device,
        in_channels,
        out_channels,
        down_position,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
