import torch
import pytest
from loguru import logger


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import AutoTokenizer
import tt_lib
from models.squeezebert.tt.squeezebert import *


@pytest.mark.parametrize(
    "model_name",
    (("squeezebert/squeezebert-uncased"),),
)
def test_gs_demo(model_name):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    question, context = (
        "Where do I live?",
        "My name is Merve and I live in İstanbul.",
    )
    inputs = tokenizer(question, context, return_tensors="pt")

    tt_attention_mask = torch_to_tt_tensor_rm(
        inputs.attention_mask, device, put_on_device=False
    )

    with torch.no_grad():
        tt_model = squeezebert_for_question_answering(device)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask, inputs.token_type_ids)

        tt_start_logits_torch = (
            tt_to_torch_tensor(tt_output.start_logits).squeeze(0).squeeze(0)
        )
        tt_end_logits_torch = (
            tt_to_torch_tensor(tt_output.end_logits).squeeze(0).squeeze(0)
        )

        answer_start_index = tt_start_logits_torch.argmax()
        answer_end_index = tt_end_logits_torch.argmax()

        predict_answer_tokens = inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]

        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    logger.info("Context: ")
    logger.info(context)
    logger.info("Question: ")
    logger.info(question)
    logger.info("GS's Predicted answer: ")
    logger.info(answer)

    tt_lib.device.CloseDevice(device)
