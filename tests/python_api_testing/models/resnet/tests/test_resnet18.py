from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")
sys.path.append(f"{f}/../tt")

from loguru import logger
import torch
from torchvision import models
import pytest
from Resnet_block import ResNet, BasicBlock, resnet18
import tt_lib

from sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc




@pytest.mark.parametrize("fold_batchnorm", [False, True], ids=['Batchnorm not folded', "Batchnorm folded"])
def test_run_resnet18_inference(fold_batchnorm, imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch.manual_seed(1234)

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        host = tt_lib.device.GetHost()

        torch_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        torch_resnet.eval()

        state_dict = torch_resnet.state_dict()

        tt_resnet18 = resnet18(device, host, fold_batchnorm)

        torch_output = torch_resnet(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_resnet18(image)

        logger.info(comp_allclose_and_pcc(torch_output, tt_output))
        passing, info = comp_pcc(torch_output, tt_output)
        logger.info(info)

        assert passing
