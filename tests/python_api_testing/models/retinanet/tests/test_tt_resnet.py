from loguru import logger
import torch
from torchvision import models
import pytest

import tt_lib
from models.retinanet.tt.resnet import TtResnet50, TtBottleneck
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_resnet50_inference(pcc, imagenet_sample_input, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    tt_resnet50 = TtResnet50(
        TtBottleneck,
        [3, 4, 6, 3],
        device=device,
        host=host,
        state_dict=torch_resnet50.state_dict(),
    )

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device)

    torch_output = torch_resnet50(imagenet_sample_input).unsqueeze(1).unsqueeze(1)
    tt_output = tt_resnet50(tt_input)

    tt_output_torch = tt_to_torch_tensor(tt_output, host)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if does_pass:
        logger.info("Resnet50 Passed!")
    else:
        logger.warning("Resnet50 Failed!")

    assert does_pass
