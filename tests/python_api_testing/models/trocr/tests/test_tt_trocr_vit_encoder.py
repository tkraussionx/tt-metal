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

from models.trocr.tt.trocr_vit_encoder import TtViTEncoder


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_trocr_vit_encoder_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        config = model.encoder.config

        # run torch model
        input = torch.rand(1, 3, 384, 384)
        head_mask = 12 * [None]
        output_attentions = False
        output_hidden_states = False
        return_dict = True

        embedding_output = model.encoder.embeddings(input)
        reference = model.encoder.encoder

        model_output = reference(
            embedding_output,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )[0]

        # run tt model
        tt_embedding_output = torch_to_tt_tensor_rm(
            embedding_output, device, put_on_device=False
        )
        tt_layer = TtViTEncoder(
            config,
            base_address="encoder.encoder",
            state_dict=model.state_dict(),
            device=device,
            host=host,
        )

        tt_output = tt_layer(
            tt_embedding_output,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )[0]

        tt_output_torch = tt_to_torch_tensor(tt_output, host)
        tt_output_torch = tt_output_torch.squeeze(0)

        # compare output
        passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

        logger.info(comp_allclose(model_output, tt_output_torch))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)
        if passing:
            logger.info("VitEncoder Passed!")
        else:
            logger.warning("VitEncoder Failed!")

        assert passing
