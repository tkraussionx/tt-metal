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
import timm
import pytest
from loguru import logger

_batch_size = 1


def test_inception_inference(imagenet_sample_input):
    image = imagenet_sample_input
    batch_size = _batch_size

    with torch.no_grad():
        torch_model = timm.create_model("inception_v4", pretrained=True)
        torch_model.eval()
        torch_output = torch_model(image)
        logger.info(torch_model.state_dict())
