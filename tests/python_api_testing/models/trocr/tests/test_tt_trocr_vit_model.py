import torch
import pytest
from loguru import logger

import pytest

from transformers import VisionEncoderDecoderModel

import tt_lib

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from tests.python_api_testing.models.utility_functions_new import (
    comp_pcc,
    comp_allclose,
)
from models.trocr.tt.trocr_vit_model import TtViTModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_vit_model_inference(pcc, reset_seeds):
    image = torch.rand(1, 3, 384, 384)
    head_mask = None
    output_attentions = None
    output_hidden_states = None
    interpolate_pos_encoding = None
    return_dict = None
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        state_dict = model.state_dict()

        reference = model.encoder

        config = model.encoder.config

        HF_output = reference(
            image,
            head_mask,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )[0]

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        tt_image = torch_to_tt_tensor_rm(image, device, put_on_device=False)
        tt_layer = TtViTModel(
            config,
            add_pooling_layer=False,
            base_address="encoder",
            state_dict=state_dict,
            device=device,
            host=host,
        )
        tt_layer.get_head_mask = reference.get_head_mask
        tt_output = tt_layer(
            tt_image,
            head_mask,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )[0]
        tt_output_torch = tt_to_torch_tensor(tt_output, host).squeeze(0)

        passing, pcc_message = comp_pcc(HF_output, tt_output_torch, pcc)

        logger.info(comp_allclose(HF_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if passing:
            logger.info("VitModel Passed!")
        else:
            logger.warning("VitModel Failed!")
