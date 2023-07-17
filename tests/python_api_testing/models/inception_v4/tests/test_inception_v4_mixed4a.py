import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import torch
from loguru import logger
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import tt_lib

from utility_functions_new import (
    comp_pcc,
    comp_allclose_and_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.inception_v4.tt.inception_v4_mixed4a import (
    TtMixed4a,
)
import torchvision.transforms as transforms
import timm


def run_test_mixed4a_inference(
    device,
    first_basic_conv2d_position,
    second_basic_conv2d_position,
    third_basic_conv2d_position,
    mixed4a_position,
    eps,
    momentum,
    pcc,
):
    # load inception v4 model =================================================
    hugging_face_reference_model = timm.create_model("inception_v4", pretrained=True)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # get BasiMixed3acConv2d module ==================================================
    Mixed4a = hugging_face_reference_model.features[mixed4a_position]
    logger.debug(f"Mixed: {Mixed4a}")

    # create input
    torch.manual_seed(0)
    # set in_channels variable (get Conv2d from BasicConv2d)
    in_channels = Mixed4a.branch0[0].conv.in_channels
    logger.debug(f"in_channels: {in_channels}")
    test_input = torch.rand(1, 160, 63, 63)  # $$

    # Pytorch call =========================================================
    pt_out = Mixed4a(test_input)
    logger.debug(f"pt_out shape: {pt_out.shape}")

    # tt call ==============================================================
    tt_module = TtMixed4a(
        device,
        hugging_face_reference_model,
        eps,
        momentum,
    )

    with torch.no_grad():
        tt_module.eval()
        test_input = torch2tt_tensor(test_input, device)
        tt_out = tt_module(test_input)
        # tt_out = tt2torch_tensor(tt_out)
        logger.debug(f"tt_out shape: {tt_out.shape}")

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test Mixed3a Passed!")
    else:
        logger.warning("test Mixed3a Failed!")

    assert does_pass


# parameters: BasicConv2d position in the model
_first_basic_conv2d_position = 3  # numbers: 0, 1, ..., 21
_second_basic_conv2d_position = "conv"  # "branch0", "branch1", "branch2", "conv", None
_third_basic_conv2d_position = None  # numbers: 0, 1, 2, 3, 4, None
_mixed4a_position = 4

# BatchNorm 2D
_eps = 0.001
_momentum = 0.1


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_mixed4a_inference(pcc):
    # set parameters
    first_basic_conv2d_position = _first_basic_conv2d_position
    second_basic_conv2d_position = _second_basic_conv2d_position
    third_basic_conv2d_position = (
        str(_third_basic_conv2d_position)
        if _third_basic_conv2d_position is not None
        else None
    )
    eps = _eps
    momentum = _momentum
    mixed4a_position = _mixed4a_position

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    run_test_mixed4a_inference(
        device,
        first_basic_conv2d_position,
        second_basic_conv2d_position,
        third_basic_conv2d_position,
        mixed4a_position,
        eps,
        momentum,
        pcc,
    )

    tt_lib.device.CloseDevice(device)
