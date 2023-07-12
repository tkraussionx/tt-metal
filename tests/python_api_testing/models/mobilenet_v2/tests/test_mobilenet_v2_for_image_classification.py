import torch
import torch.nn as nn
from loguru import logger
from transformers import (
    AutoImageProcessor,
    MobileNetV2ForImageClassification,
    MobileNetV2Config,
)
from datasets import load_dataset
from models.mobilenet_v2.tt.mobilenet_v2_for_image_classification import (
    TtMobileNetV2ForImageClassification,
)
import tt_lib
from tests.python_api_testing.models.utility_functions_new import comp_pcc
from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt2torch_tensor,
)


def test_mobilenetv2_for_image_classification():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # Get model and img processor
    image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    reference_model = MobileNetV2ForImageClassification.from_pretrained(
        "google/mobilenet_v2_1.0_224"
    )
    tt_model = TtMobileNetV2ForImageClassification(
        state_dict=reference_model.state_dict(),
        device=device,
        config=reference_model.config,
    )

    # Get data
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    with torch.no_grad():
        inputs = image_processor(image, return_tensors="pt")
        torch_logits = reference_model(inputs["pixel_values"]).logits
        tt_im = torch_to_tt_tensor_rm(
            inputs["pixel_values"], device, put_on_device=False
        )
        tt_logits = tt_model(tt_im).logits

    # model predicts one of the 1000 ImageNet classes
    torch_predicted_label = torch_logits.argmax(-1).item()
    tt_logits = tt2torch_tensor(tt_logits)
    tt_lib.device.CloseDevice(device)
    tt_predicted_label = tt_logits.argmax(-1).item()

    does_pass, pcc_message = comp_pcc(torch_logits, tt_logits, 0.96)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Mobilenet V2 For Image Classification Model Passed!")
    else:
        logger.warning("Mobilenet V2 For Image Classification Model  Failed!")

    assert does_pass
