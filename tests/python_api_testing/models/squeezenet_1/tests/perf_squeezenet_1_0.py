import math
from pathlib import Path
import sys
import os

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import torch
from loguru import logger
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from PIL import Image
import tt_lib

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report, profiler

from python_api_testing.models.squeezenet_1.tt.squeezenet_1 import squeezenet_1_0
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from torchvision import transforms

BATCH_SIZE = 1


def test_perf():
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "SqueezeNet1_0_Weights.DEFAULT weights are used"

    # load squeezenet model ================================================
    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # create input =========================================================
    input_path = os.path.join(
        "tests/python_api_testing/models/squeezenet_1/demo", "input_image.jpg"
    )
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

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    tt_model = squeezenet_1_0(device, hugging_face_reference_model, state_dict)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = hugging_face_reference_model(input_batch)[0]
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(input_batch)[0]
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_model(input_batch)[0]
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        "SqueezeNet_1.0",
        BATCH_SIZE,
        first_iter_time,
        second_iter_time,
        comments,
        cpu_time,
    )
