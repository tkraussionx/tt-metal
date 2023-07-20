from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import tt_lib
import torch
import pytest
from loguru import logger
import urllib
from python_api_testing.models.inception_v4.tt.inception_v4_model import inception_v4
import timm
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)

_batch_size = 1


def test_inception_inference(imagenet_sample_input):
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    image = imagenet_sample_input
    tt_module = inception_v4(device)
    tt_module.eval()

    with torch.no_grad():
        image = torch2tt_tensor(image, device)
        tt_out = tt_module(image)
        tt_out_torch = tt2torch_tensor(tt_out).squeeze(0).squeeze(0)

        probabilities = torch.nn.functional.softmax(tt_out_torch[0], dim=0)

        # url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "tests/python_api_testing/models/inception_v4/imagenet_classes.txt")
        # urllib.request.urlretrieve(url, filename)
        filename = "tests/python_api_testing/models/inception_v4/imagenet_classes.txt"
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Print top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            logger.info(categories[top5_catid[i]], top5_prob[i].item())
