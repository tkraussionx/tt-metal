from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
from loguru import logger
import tt_lib
from python_api_testing.models.inception_v4.tt.inception_v4_model import inception_v4
import timm
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_model_inference(imagenet_sample_input):
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    reference_model = timm.create_model("inception_v4", pretrained=True)
    reference_model.eval()

    tt_module = inception_v4(device)
    tt_module.eval()

    with torch.no_grad():
        image = imagenet_sample_input
        pt_out = reference_model(image)

        image = torch2tt_tensor(image, device)
        tt_out = tt_module(image)
        tt_out_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_torch, 0.99)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("TtInceptionV4 Passed!")
    else:
        logger.warning("TtInceptionV4 Failed!")

    assert does_pass
