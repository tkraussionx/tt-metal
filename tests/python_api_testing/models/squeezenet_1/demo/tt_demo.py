import os
import sys
import pytest
import torch
import random
from loguru import logger
import numpy as np
from torch import nn
from pathlib import Path
from PIL import Image

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from torchvision import transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from python_api_testing.models.squeezenet_1.tt.squeezenet_1 import TtSqueezeNet
from python_api_testing.models.squeezenet_1.squeezenet_utils import download_image
import tt_lib


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # SqueezeNet root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def run_squeezenet_demo(device):
    random.seed(42)
    torch.manual_seed(42)

    source = str(ROOT)
    download_image(Path(source))

    input_path = os.path.join(Path(source), "input_image.jpg")
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

    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    state_dict = hugging_face_reference_model.state_dict()

    # tt call ======================================================================
    tt_module = TtSqueezeNet(device, hugging_face_reference_model, state_dict)
    tt_module.eval()

    output = tt_module(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open(os.path.join(Path(source), "imagenet_classes.txt"), "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()

    logger.info(f"Grayskull's predicted Output: {result}")


def test_gs_demo():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_squeezenet_demo(device)

    tt_lib.device.CloseDevice(device)
