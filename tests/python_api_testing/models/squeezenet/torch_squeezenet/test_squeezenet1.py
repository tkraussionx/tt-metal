
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torchvision import models
import pytest
from loguru import logger
from libs import tt_lib as ttl
from tqdm import tqdm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc, comp_pcc
from squeezenet import squeezenet1_0


_batch_size = 1

@pytest.mark.parametrize("fuse_ops", [False, True], ids=['Not Fused', 'Ops Fused'])
def test_squeezenet1_inference(fuse_ops, imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()
        torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)

        torch_squeezenet.eval()

        state_dict = torch_squeezenet.state_dict()
        if not fuse_ops:
            tt_squeezenet = squeezenet1_0(device=device, host=host, disable_conv_on_tt_device=fuse_ops)
        else:
            tt_squeezenet = squeezenet1_0(device=None, host=None, disable_conv_on_tt_device=fuse_ops)
        tt_squeezenet.eval()



        torch_output = torch_squeezenet(image).unsqueeze(1).unsqueeze(1)
        tt_output = tt_squeezenet(image)

        passing = comp_pcc(torch_output, tt_output)

        assert passing[0], passing[1:]

    logger.info(f"PASSED {passing[1]}")
