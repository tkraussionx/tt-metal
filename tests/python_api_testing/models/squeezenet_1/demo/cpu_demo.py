import os
import sys
import random
import pytest
import torch
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
from torchvision.models import (
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights,
)


def test_cpu_demo(model_location_generator):
    random.seed(42)
    torch.manual_seed(42)

    data_path = model_location_generator("tt_dnn-models/SqueezeNet/data/")
    data_image_path = str(data_path / "images")

    # Read the categories
    with open(os.path.join(data_path, "imagenet_classes.txt"), "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # make prediction for all images from weka folder
    for img in os.listdir(data_image_path):
        # load image
        input_path = os.path.join(data_image_path, img)
        input_image = Image.open(input_path)

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        with torch.no_grad():
            # Select PyTorch model
            torch_squeezenet = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
            # torch_squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
            output = torch_squeezenet(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        result = {}
        for i in range(top5_prob.size(0)):
            result[categories[top5_catid[i]]] = top5_prob[i].item()

        logger.info(f"CPU's classification for {img}: {result}")
