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
from python_api_testing.models.squeezenet_1.reference.squeezenet import SqueezeNet
from python_api_testing.models.squeezenet_1.tt.squeezenet_1 import TtSqueezeNet
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights


def run_test_squeezenet_inference(device, pcc):
    # load squeezenet model ================================================
    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 64, 64)

    pt_out = hugging_face_reference_model(test_input)
    logger.debug(f"pt_out shape {pt_out.shape}")

    # PtSqueezeNet = SqueezeNet(state_dict)
    # PtSqueezeNet.eval()
    # tt_out = PtSqueezeNet(test_input)
    # logger.debug(f"pt_out2 shape {tt_out.shape}")

    # tt call ==============================================================
    tt_module = TtSqueezeNet(device, hugging_face_reference_model, state_dict)
    tt_module.eval()

    # CHANNELS_LAST
    tt_out = tt_module(test_input)

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_Squeezenet Passed!")
    else:
        logger.warning("test_Squeezenet Failed!")

    assert does_pass


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_squeezenet_inference(pcc):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_squeezenet_inference(device, pcc)
    tt_lib.device.CloseDevice(device)
