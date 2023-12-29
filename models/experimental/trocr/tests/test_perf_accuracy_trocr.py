# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import evaluate

from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import tt_lib
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.trocr.tt.trocr import trocr_causal_llm
import xml.etree.ElementTree as ET
import random

BATCH_SIZE = 1


def get_images(model_location_generator, rand_seed, dataset_base_addr):
    random.seed(rand_seed)
    label_xml = dataset_base_addr + "/word.xml"
    tree = ET.parse(label_xml)
    root = tree.getroot()
    images = []

    # Access elements and attributes
    for image in root.findall("image"):
        rel_path = image.attrib["file"]
        tag = image.attrib["tag"]
        images.append((rel_path, tag))

    random.shuffle(images)
    return images


def run_perf(model_location_generator, expected_inference_time, expected_compile_time, iterations, rand_seed, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values
    dataset_base_addr = str(model_location_generator("trocr/ICDAR_2003"))

    images = get_images(model_location_generator, rand_seed, dataset_base_addr)

    tt_model = trocr_causal_llm(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = model.generate(pixel_values)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model.generate(pixel_values)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model.generate(pixel_values)
        profiler.end(second_key)

        golden_labels = []
        generated_labels = []
        profiler.start(third_key)
        for i in range(iterations):
            img_input = Image.open(dataset_base_addr + "/" + images[i][0])
            pixel_values = processor(images=img_input, return_tensors="pt").pixel_values
            tt_output = tt_model.generate(pixel_values)
            generated_text_tt = processor.batch_decode(tt_output, skip_special_tokens=True)[0]
            generated_labels.append(generated_text_tt)
            golden_labels.append(images[i][1])
            logger.info("TrOCR Model answered")
            logger.info(generated_text_tt)
        profiler.end(third_key)

        cer = evaluate.load("cer")
        cer_score_tt = cer.compute(predictions=generated_labels, references=golden_labels)

        first_iter_time = profiler.get(first_key)
        second_iter_time = profiler.get(second_key)
        third_iter_time = profiler.get(third_key)
        cpu_time = profiler.get(cpu_key)

        prep_perf_report(
            "trocr",
            BATCH_SIZE,
            first_iter_time,
            second_iter_time,
            expected_compile_time,
            expected_inference_time,
            "perf_accuracy",
            cpu_time,
        )
        compile_time = first_iter_time - second_iter_time

        logger.info(f"trocr inference time: {second_iter_time}")
        logger.info(f"trocr compile time: {compile_time}")
        logger.info(f"trocr inference for {iterations} samples: {third_iter_time}")
        logger.info(f"cer_score: {cer_score_tt}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations, rand_seed",
    ((199.57233452796936, 16.2, 1, 42),),
)
def test_perf_bare_metal(
    use_program_cache,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    rand_seed,
    device,
):
    run_perf(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        rand_seed,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations, rand_seed",
    ((199.57233452796936, 16.2, 1, 42),),
)
def test_perf_virtual_machine(
    use_program_cache,
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    rand_seed,
    device,
):
    run_perf(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        rand_seed,
        device,
    )
