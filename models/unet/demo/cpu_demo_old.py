import pytest
import torch
from torch import nn
from loguru import logger
import random
from PIL import Image


def test_cpu_demo():
    random.seed(42)
    torch.manual_seed(42)

    # ======================================================================================================
    UNet = torch.hub.load(
        "milesial/Pytorch-UNet", "unet_carvana", pretrained=False, scale=0.5
    )
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )

    UNet.load_state_dict(checkpoint)

    # print(UNet.state_dict())

    input_image = Image.open("models/unet/bmw.jpeg")

    Output = UNet(input_image)
