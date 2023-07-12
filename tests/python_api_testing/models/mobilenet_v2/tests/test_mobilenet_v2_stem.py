import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from transformers import AutoImageProcessor, MobileNetV2Model, MobileNetV2Config
from datasets import load_dataset
from models.mobilenet_v2.tt.mobilenet_v2_stem import (
    TtMobileNetV2Stem,
)
import tt_lib
from tests.python_api_testing.models.utility_functions_new import comp_pcc
from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt2torch_tensor,
)


def test_stem_layer():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get data
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Get model and img processor
    image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    reference_model = MobileNetV2Model.from_pretrained(
        "google/mobilenet_v2_1.0_224"
    )  # load FP32 model

    base_address = f"conv_stem"
    torch_model = reference_model.conv_stem

    # Get input params
    in_channels = torch_model.first_conv.convolution.in_channels
    out_channels = torch_model.reduce_1x1.convolution.out_channels
    expanded_channels = torch_model.first_conv.convolution.out_channels

    tt_model = TtMobileNetV2Stem(
        base_address=base_address,
        state_dict=reference_model.state_dict(),
        device=device,
        config=reference_model.config,
        in_channels=in_channels,
        out_channels=out_channels,
        expanded_channels=expanded_channels,
    )

    # Run inference
    with torch.no_grad():
        tt_model.eval()
        torch_model.eval()

        inputs = image_processor(image, return_tensors="pt")["pixel_values"]
        pt_out = torch_model(inputs)
        tt_im = torch_to_tt_tensor_rm(inputs, device, put_on_device=False)
        tt_out = tt_model(tt_im)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Mobilenet V2 Stem Layer Passed!")
    else:
        logger.warning("Mobilenet V2 Stem Layer Failed!")

    assert does_pass
