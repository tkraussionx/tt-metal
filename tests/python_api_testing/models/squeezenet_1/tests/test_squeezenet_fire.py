import pytest
import torch
from loguru import logger
from torch import nn
import tt_lib

from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from tests.python_api_testing.models.utility_functions_new import (
    comp_pcc,
    comp_allclose_and_pcc,
)
from models.squeezenet.tt.squeezenet_fire import TtFire
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights


def run_test_fire_inference(device, fire_position, pcc):
    # load squeezenet model ================================================
    hugging_face_reference_model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    # get Fire module =====================================================
    FireBlock = hugging_face_reference_model.features[fire_position]

    _inplanes = FireBlock.squeeze.in_channels
    _squeeze_planes = FireBlock.squeeze.out_channels
    _expand1x1_planes = FireBlock.expand1x1.out_channels
    _expand3x3_planes = FireBlock.expand3x3.out_channels

    logger.debug(f"inplanes: {_inplanes}")
    logger.debug(f"squeeze_planes: {_squeeze_planes}")
    logger.debug(f"expand1x1_planes: {_expand1x1_planes}")
    logger.debug(f"expand3x3_planes: {_expand3x3_planes}")

    # create input
    torch.manual_seed(0)
    test_input = torch.rand(1, _inplanes, 64, 64)

    # Pytorch call =========================================================
    pt_out = FireBlock(test_input)

    # tt call ==============================================================
    tt_module = TtFire(
        device,
        hugging_face_reference_model,
        fire_position,
        inplanes=_inplanes,
        squeeze_planes=_squeeze_planes,
        expand1x1_planes=_expand1x1_planes,
        expand3x3_planes=_expand3x3_planes,
    )

    # CHANNELS_LAST
    test_input = torch2tt_tensor(test_input, device)
    tt_out = tt_module(test_input)

    logger.debug(f"pt_out shape {pt_out.shape}")
    logger.debug(f"tt_out shape {tt_out.shape}")

    _, comp_out = comp_allclose_and_pcc(pt_out, tt_out)
    logger.info(comp_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test Squeezenet 1.0 Fire Passed!")
    else:
        logger.warning("test Squeezenet 1.0 Fire Failed!")

    assert does_pass


# parameters: Fire position in the model
_fire_position = 4


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_fire_inference(pcc):
    # Initialize the device
    fire_position = _fire_position

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    host = tt_lib.device.GetHost()

    run_test_fire_inference(device, fire_position, pcc)
    tt_lib.device.CloseDevice(device)
