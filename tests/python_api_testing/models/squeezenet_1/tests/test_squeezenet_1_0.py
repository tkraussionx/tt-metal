import math
from pathlib import Path
import sys
import os

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
from PIL import Image

from utility_functions_new import (
    comp_pcc,
    comp_allclose_and_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.squeezenet_1.tt.squeezenet_1 import squeezenet_1_0
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from python_api_testing.models.squeezenet_1.squeezenet_utils import download_image


def run_test_squeezenet_inference(device, pcc):
    # load squeezenet model ================================================
    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # take input
    download_image("tests/python_api_testing/models/squeezenet_1/demo")

    input_path = os.path.join(
        "tests/python_api_testing/models/squeezenet_1/demo", "input_image.jpg"
    )
    input_image = Image.open(input_path)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # Pytorch call =========================================================
    pt_out = hugging_face_reference_model(input_batch)
    logger.debug(f"pt_out shape {pt_out.shape}")

    # tt call ==============================================================
    tt_module = squeezenet_1_0(device, hugging_face_reference_model, state_dict)
    tt_module.eval()

    # CHANNELS_LAST
    tt_out = tt_module(input_batch)

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
