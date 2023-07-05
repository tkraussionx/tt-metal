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

from python_api_testing.models.yolov7.tt.yolov7_sppcspc import TtSPPCSPC
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


def test_sppcspc_module(model_location_generator):
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

    INDEX = 51
    base_address = f"model.{INDEX}"
    torch_model = reference_model.model[INDEX]

    tt_model = TtSPPCSPC(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        c1=1024,
        c2=512,
    )

    # Run inference
    with torch.no_grad():
        im = torch.rand(1, 1024, 32, 32)

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
        logger.info("YOLOv7 TtSPPCSPC Passed!")
    else:
        logger.warning("YOLOv7 TtSPPCSPC Failed!")

    assert does_pass
