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

from models.unet.tt.unet_model import TtUnet
from tests.python_api_testing.models.unet.reference.unet_model import UNet


def run_test_unet_inference(
    device,
    n_channels,
    n_classes,
    pcc,
):
    # load Unet model ================================================
    reference_model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )
    reference_model.load_state_dict(checkpoint)
    reference_model.eval()
    state_dict = reference_model.state_dict()

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 256, 256)

    # get Pytorch subgraph ==========================================
    logger.debug(f"CPU model: {reference_model}")

    # Pytorch call
    pt_out = reference_model(test_input)
    logger.debug(f"pt_out shape {pt_out.shape}")

    # get TtUnet module ========================================
    n_channels = 3
    n_classes = 2

    gs_module = TtUnet(device, state_dict, n_channels, n_classes, False)

    # TT call ========================================================
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
        logger.info("test Unet Passed!")
    else:
        logger.warning("test Unet Failed!")

    assert does_pass


# parameters: Unet model
_n_channels = 3
_n_classes = 2


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_unet_inference(pcc):
    # Initialize the device
    n_channels = _n_channels
    n_classes = _n_classes

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_unet_inference(
        device,
        n_channels,
        n_classes,
        pcc,
    )
    tt_lib.device.CloseDevice(device)
