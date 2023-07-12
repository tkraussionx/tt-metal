import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from transformers import (
    MobileNetV2Model,
    MobileNetV2ForImageClassification,
    MobileNetV2Config,
)
from models.mobilenet_v2.tt.mobilenet_v2_inverted_residual import (
    TtMobileNetV2InvertedResidual,
)
import tt_lib
from tests.python_api_testing.models.utility_functions_new import comp_pcc
from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt2torch_tensor,
)


def test_inverted_residual_layer():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get model
    reference_model = MobileNetV2Model.from_pretrained(
        "google/mobilenet_v2_1.0_224"
    )  # load FP32 model

    BLOCK = 0
    base_address = f"layer.{BLOCK}"
    torch_model = reference_model.layer[BLOCK]

    # Get input params
    in_channels = torch_model.expand_1x1.convolution.in_channels
    out_channels = torch_model.reduce_1x1.convolution.out_channels
    stride = torch_model.conv_3x3.convolution.stride[0]
    dilation = torch_model.conv_3x3.convolution.dilation[0]

    tt_model = TtMobileNetV2InvertedResidual(
        base_address=base_address,
        state_dict=reference_model.state_dict(),
        device=device,
        config=reference_model.config,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        dilation=dilation,
    )

    # Get data
    inputs = torch.rand([1, 16, 112, 112])

    # Run inference
    with torch.no_grad():
        torch_model.eval()
        tt_model.eval()

        pt_out = torch_model(inputs)

        tt_im = torch2tt_tensor(
            inputs, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        tt_out = tt_model(tt_im)

    tt_out = tt2torch_tensor(tt_out)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Mobilenet V2 Inverted Residual Passed!")
    else:
        logger.warning("Mobilenet V2 Inverted Residual Failed!")

    assert does_pass
