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

from python_api_testing.models.yolov7.tt.yolov7_detect import TtDetect
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


def test_detect_module(model_location_generator):
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Get data
    data_path = model_location_generator("tt_dnn-models/Yolo/data/")

    # Load model
    model_path = model_location_generator("tt_dnn-models/Yolo/models/")
    weights = str(model_path / "yolov7.pt")
    reference_model = get_yolov7_fused_cpu_model(
        model_location_generator
    )  # load FP32 model

    state_dict = reference_model.state_dict()

    INDEX = 105
    base_address = f"model.{INDEX}"
    torch_model = reference_model.model[INDEX]

    nc = 80
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    ch = [128, 256, 512]

    torch_model = reference_model.model[INDEX]

    tt_model = TtDetect(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        nc=nc,
        anchors=anchors,
        ch=ch,
    )
    tt_model.anchors = torch_model.anchors
    tt_model.stride = torch.tensor([8.0, 16.0, 32.0])

    a = torch.rand(1, 256, 64, 80)
    b = torch.rand(1, 512, 32, 40)
    c = torch.rand(1, 1024, 16, 20)
    test_input = [a, b, c]

    with torch.no_grad():
        torch_model.eval()
        pt_out = torch_model(test_input)

    tt_a = torch2tt_tensor(a, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_b = torch2tt_tensor(b, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_c = torch2tt_tensor(c, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_test_input = [tt_a, tt_b, tt_c]

    with torch.no_grad():
        tt_model.eval()
        tt_out = tt_model(tt_test_input)

    tt_lib.device.CloseDevice(device)

    does_all_pass = True

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out[0], 0.99)
    does_all_pass &= does_pass
    logger.info(f"Output prediction from the highest scale: {pcc_message}")

    for i in range(len(pt_out[1])):
        does_pass, pcc_message = comp_pcc(pt_out[1][i], tt_out[1][i], 0.99)
        logger.info(f"Object detection {i}: {pcc_message}")
        does_all_pass &= does_pass

    if does_all_pass:
        logger.info(f"Yolov7 Detection Head Passed!")
    else:
        logger.warning(f"Yolov7 Detection Head Failed!")

    assert does_all_pass
