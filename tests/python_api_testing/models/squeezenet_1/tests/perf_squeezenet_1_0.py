import os
import pytest
import torch
from loguru import logger
from torch import nn
from PIL import Image
import tt_lib

from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)

from tests.python_api_testing.models.utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    prep_report,
)

from models.squeezenet.tt.squeezenet_1 import squeezenet_1_0
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from torchvision import transforms
from models.squeezenet.squeezenet_utils import download_image

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

    # load image
    data_path = "tests/python_api_testing/models/squeezenet_1"
    download_image(data_path)

    image_name = "dog.jpg"
    input_path = os.path.join(data_path, image_name)
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
