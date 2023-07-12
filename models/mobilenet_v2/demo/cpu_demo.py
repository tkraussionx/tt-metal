import tt_lib
from transformers import (
    AutoImageProcessor,
    MobileNetV2ForImageClassification,
)
from datasets import load_dataset
import torch
from loguru import logger


def test_cpu_demo():
    torch.manual_seed(1234)

    # Get data
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Get model
    image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    model = MobileNetV2ForImageClassification.from_pretrained(
        "google/mobilenet_v2_1.0_224"
    )

    with torch.no_grad():
        inputs = image_processor(image, return_tensors="pt")
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    logger.info(
        f"CPU Model MobileNetV2ForImageClassification predicted label: {model.config.id2label[predicted_label]}"
    )
