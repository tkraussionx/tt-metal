import torch
import torch.nn as nn
from loguru import logger
from models.mobilenet_v2.mobilenet_v2_mini_graphs import (
    TtIdentity,
)
import tt_lib
from tests.python_api_testing.models.utility_functions_new import comp_pcc
from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt2torch_tensor,
)


def test_identity():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
    inputs = torch.randn(1, 1, 128, 20)
    torch_output = m(inputs)

    tt_inputs = torch_to_tt_tensor_rm(inputs, device, put_on_device=False)
    tt_m = TtIdentity(54, unused_argument1=0.1, unused_argument2=False)
    logger.info(tt_m)
    tt_output = tt_m(tt_inputs)

    tt_output = tt2torch_tensor(tt_output)
    tt_lib.device.CloseDevice(device)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Identity operation Passed!")
    else:
        logger.warning("Identity operation Failed!")

    assert does_pass
