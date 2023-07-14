import torch
import pytest
from loguru import logger

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

from models.trocr.tt.trocr_vit_self_attention import TtViTSelfAttention


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_vit_self_attention_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        config = model.encoder.config

        base_address = f"encoder.encoder.layer.0.attention.attention"

        torch_model = model.encoder.encoder.layer[0].attention.attention

        tt_model = TtViTSelfAttention(
            config=config,
            base_address=base_address,
            state_dict=model.state_dict(),
            device=device,
            host=host,
        )

        # run torch model
        input = torch.rand(1, 1, 197, 768).squeeze(0)
        head_mask = None
        output_attentions = False
        model_output = torch_model(input, head_mask, output_attentions)[0]

        # run tt model
        tt_input = torch_to_tt_tensor_rm(input, host)
        tt_output = tt_model(tt_input, head_mask, output_attentions)[0]
        tt_output_torch = tt_to_torch_tensor(tt_output, host)
        tt_output_torch = tt_output_torch.squeeze(0)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if passing:
            logger.info("VitSelfAttention Passed!")
        else:
            logger.warning("VitSelfAttention Failed!")

        assert passing
