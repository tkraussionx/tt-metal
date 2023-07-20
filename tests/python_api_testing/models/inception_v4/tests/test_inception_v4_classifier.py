from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
from loguru import logger
import tt_lib
from python_api_testing.models.inception_v4.tt.inception_v4_classifier import (
    TtClassifier,
)
import timm
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_classifier_inference(imagenet_sample_input):
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    reference_model = timm.create_model("inception_v4", pretrained=True)
    reference_model.eval()

    tt_module = TtClassifier(
        device=device,
        state_dict=reference_model.state_dict(),
        num_classes=reference_model.num_classes,
        num_features=reference_model.num_features,
        pool_type="avg",
    )
    tt_module.eval()

    with torch.no_grad():
        test_input = torch.rand(1, 1536, 64, 64)
        pt_out = reference_model.forward_head(test_input)

        test_input = torch2tt_tensor(test_input, device)
        tt_out = tt_module(test_input)
        tt_out_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_torch, 0.99)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("TtClassifier Passed!")
    else:
        logger.warning("TtClassifier Failed!")

    assert does_pass
