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


def test_inception_inference(imagenet_sample_input):
    image = imagenet_sample_input

    with torch.no_grad():
        torch_model = timm.create_model("inception_v4", pretrained=True)
        torch_model.eval()
        torch_output = torch_model(image)

        probabilities = torch.nn.functional.softmax(torch_output[0], dim=0)

        filename = "tests/python_api_testing/models/inception_v4/imagenet_classes.txt"
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Print top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            logger.info(categories[top5_catid[i]], top5_prob[i].item())
