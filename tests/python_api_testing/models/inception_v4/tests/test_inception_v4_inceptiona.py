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
from python_api_testing.models.inception_v4.tt.inception_v4_inceptiona import (
    TtInceptionA,
)
import timm
from utility_functions_new import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
)


def test_inceptiona_inference():
    torch.manual_seed(1234)

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    reference_model = timm.create_model("inception_v4", pretrained=True)
    reference_model.eval()

    block_id = 6
    torch_model = reference_model.features[block_id]
    base_address = f"features.{block_id}"

    tt_module = TtInceptionA(
        device=device,
        state_dict=reference_model.state_dict(),
        base_address=base_address,
    )
    tt_module.eval()
    torch_model.eval()

    with torch.no_grad():
        test_input = torch.rand(1, 384, 64, 64)
        pt_out = torch_model(test_input)

        test_input = torch2tt_tensor(test_input, device)
        tt_out = tt_module(test_input)
        tt_out_torch = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_torch, 0.99)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("TtInceptionA Passed!")
    else:
        logger.warning("TtInceptionA Failed!")

    assert does_pass
