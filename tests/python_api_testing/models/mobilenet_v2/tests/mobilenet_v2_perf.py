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
from tests.python_api_testing.models.utility_functions_new import (
    profiler,
    prep_report,
    comp_pcc,
)
from models.utility_functions import (
    torch2tt_tensor,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.utility_functions import disable_compile_cache, enable_compile_cache

BATCH_SIZE = 1


def test_perf():
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "mobilenet_v2_1.0_224"

    # Initialize the device
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
        # Prepare inputs
        inputs = image_processor(image, return_tensors="pt")
        tt_im = torch_to_tt_tensor_rm(
            inputs["pixel_values"], device, put_on_device=False
        )

        profiler.start(cpu_key)
        torch_logits = reference_model(inputs["pixel_values"]).logits
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_logits = tt_model(tt_im).logits
        profiler.end(first_key)

        profiler.start(second_key)
        tt_logits = tt_model(tt_im).logits
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    tt_lib.device.CloseDevice(device)

    prep_report(
        "mobilenet_v2",
        BATCH_SIZE,
        first_iter_time,
        second_iter_time,
        comments,
        cpu_time,
    )
