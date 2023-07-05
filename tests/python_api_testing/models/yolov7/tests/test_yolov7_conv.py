import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.yolov7.reference.utils.datasets import LoadImages
from python_api_testing.models.yolov7.reference.utils.general import check_img_size
from python_api_testing.models.yolov7.reference.models.yolo import Conv
from python_api_testing.models.yolov7.tt.yolov7_conv import TtConv
from python_api_testing.models.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
import tt_lib
from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


def test_conv_module(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get data
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    data_image_path = str(data_path / "images/horses.jpg")

    imgsz = 640
    save_img = True  # save inference images

    # Load model
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    weights = str(model_path / "yolov7.pt")
    reference_model = get_yolov7_fused_cpu_model(
        model_location_generator
    )  # load FP32 model

    state_dict = reference_model.state_dict()

    INDEX = 0
    base_address = f"model.{INDEX}"
    torch_model = reference_model.model[INDEX]

    in_channels = torch_model.conv.in_channels
    out_channels = torch_model.conv.out_channels
    kernel_size = torch_model.conv.kernel_size[0]
    stride = torch_model.conv.stride[0]
    padding = torch_model.conv.padding[0]
    groups = torch_model.conv.groups
    dilation = torch_model.conv.dilation[0]

    tt_model = TtConv(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=in_channels,
        c2=out_channels,
        k=kernel_size,
        s=stride,
        p=padding,
        g=groups,
        d=dilation,
    )

    # Load data
    stride = max(int(max(torch_model.conv.stride)), 32)  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride)

    real_input = True

    # Run inference
    with torch.no_grad():
        if real_input:
            path, im, _, _ = next(iter(dataset))
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

        else:
            im = torch.rand(1, 3, 640, 640)

        # Inference- fused
        pt_out = torch_model(im)

        tt_im = torch_to_tt_tensor_rm(im, device)
        tt_out = tt_model(tt_im)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_output_torch)

    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("YOLOv7 TtConv Passed!")
    else:
        logger.warning("YOLOv7 TtConv Failed!")

    assert does_pass
