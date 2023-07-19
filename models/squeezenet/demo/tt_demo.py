import os
import pytest
import torch
from torch import nn
import random
from loguru import logger
from PIL import Image

from torchvision import transforms
from torchvision.models import (
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights,
)
from models.squeezenet.tt.squeezenet_1 import (
    squeezenet_1_0,
    squeezenet_1_1,
)
import tt_lib


def run_squeezenet_demo(device, model_location_generator):
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

        # SELECT PyTorch MODEL ================================================================
        hugging_face_reference_model = squeezenet1_0(
            weights=SqueezeNet1_0_Weights.DEFAULT
        )
        # hugging_face_reference_model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        state_dict = hugging_face_reference_model.state_dict()

        # SELECT TT MODEL =====================================================================
        tt_module = squeezenet_1_0(device, hugging_face_reference_model, state_dict)
        # tt_module = squeezenet_1_1(device, hugging_face_reference_model, state_dict)
        tt_module.eval()

        output = tt_module(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        result = {}
        for i in range(top5_prob.size(0)):
            result[categories[top5_catid[i]]] = top5_prob[i].item()

        logger.info(f"Grayskull's classification for {img}: {result}")


def test_gs_demo(model_location_generator):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_squeezenet_demo(device, model_location_generator)

    tt_lib.device.CloseDevice(device)
