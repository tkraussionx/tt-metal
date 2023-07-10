from loguru import logger
import torch
from torchvision import models
import pytest

import tt_lib
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm
from models.retinanet.tt.intermediate_layer import (
    TtIntermediateLayerGetter,
    Bottleneck,
)
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_intermediate_layer_inference(pcc, imagenet_sample_input, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    base_address = "backbone.body."

    model = models.detection.retinanet_resnet50_fpn_v2(
        weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights
    )

    torch_model = model.backbone.body
    tt_model = TtIntermediateLayerGetter(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        host=host,
        state_dict=model.state_dict(),
        base_address=base_address,
    )

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device)

    torch_outputs = list(torch_model(imagenet_sample_input).values())
    tt_outputs = tt_model(tt_input)
    tt_outputs_torch = [tt_to_torch_tensor(res, host) for res in tt_outputs]

    does_pass_list = []
    for i in range(len(tt_outputs_torch)):
        does_pass, pcc_message = comp_pcc(torch_outputs[i], tt_outputs_torch[i], pcc)
        does_pass_list.append(does_pass)
        logger.info(comp_allclose(torch_outputs[i], tt_outputs_torch[i]))
        logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if all(does_pass_list):
        logger.info("IntermediateLayerGetter Passed!")
    else:
        logger.warning("IntermediateLayerGetter Failed!")

    assert all(does_pass_list)
