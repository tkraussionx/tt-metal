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
from python_api_testing.models.inception_v4.tt.inception_v4_mixed3a import (
    TtMixed3a,
)
import torchvision.transforms as transforms
import timm


def run_test_mixed3a_inference(device, pcc):
    # load inception v4 model =================================================
    hugging_face_reference_model = timm.create_model("inception_v4", pretrained=True)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # get BasiMixed3acConv2d module ==================================================
    mixed3a_position = 3
    Mixed3a = hugging_face_reference_model.features[mixed3a_position]

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 64, 128, 128)

    # Pytorch call =========================================================
    pt_out = Mixed3a(test_input)
    logger.debug(f"pt_out shape: {pt_out.shape}")

    # tt call ==============================================================
    tt_module = TtMixed3a(device, hugging_face_reference_model, mixed3a_position)

    # CHANNELS_LAST
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)
    tt_out = tt2torch_tensor(tt_out)
    logger.debug(f"tt_out shape: {tt_out.shape}")

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test BasicConv2d Passed!")
    else:
        logger.warning("test BasicConv2d Failed!")

    assert does_pass


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_mixed3a_inference(pcc):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    run_test_mixed3a_inference(device, pcc)

    tt_lib.device.CloseDevice(device)
