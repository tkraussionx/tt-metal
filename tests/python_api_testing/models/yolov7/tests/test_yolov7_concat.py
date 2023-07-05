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

from python_api_testing.models.yolov7.reference.models.yolo import Concat
from python_api_testing.models.yolov7.tt.yolov7_concat import TtConcat
from python_api_testing.models.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
import tt_lib
from utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_concat_module(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # Load model
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    weights = str(model_path / "yolov7.pt")
    reference_model = get_yolov7_fused_cpu_model(
        model_location_generator
    )  # load FP32 model

    state_dict = reference_model.state_dict()

    INDEX = 10
    base_address = f"model.{INDEX}"

    torch_model = reference_model.model[INDEX]

    tt_model = TtConcat(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
    )

    # Create random Input image with channels > 3
    im_1 = torch.rand(1, 64, 512, 640)
    im_2 = torch.rand(1, 64, 512, 640)
    # Inference
    pred = torch_model([im_1, im_2])

    tt_im_1 = torch2tt_tensor(im_1, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_im_2 = torch2tt_tensor(im_2, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    x = [tt_im_1, tt_im_2]
    tt_pred = tt_model(x)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_pred)

    does_pass, pcc_message = comp_pcc(pred, tt_output_torch)
    logger.info(pcc_message)
    _, comp_out = comp_allclose_and_pcc(pred, tt_output_torch)
    logger.info(comp_out)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("YOLOv7 TtConcat Passed!")
    else:
        logger.warning("YOLOv7 TtConcat Failed!")

    assert does_pass
