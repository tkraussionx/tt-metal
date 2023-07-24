from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from datasets import load_dataset
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from tests.python_api_testing.models.conftest import model_location_generator_
import pytest
import tt_lib as ttl
from utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from test_bert_batch_dram import TtBertBatchDram

from utility_functions import (
    enable_compile_cache,
    enable_compilation_reports,
    enable_memory_reports,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_compile_cache,
    profiler,
)
from utility_functions_new import prep_report
import pytest
from loguru import logger

BATCH_SIZE = 8
model_name = "phiyodr/bert-large-finetuned-squad2"
tokenizer_name = "phiyodr/bert-large-finetuned-squad2"
comments = "Large"
seq_len = 384
real_input = True
attention_mask = True
token_type_ids = True
model_location_generator = model_location_generator_

MIXED_PRECISION_BATCH8_MODEL_CONFIG = {
    "DEFAULT_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "DEFAULT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),  # Needs to be DRAM
    "OP1_FUSED_QKV_MM_BIAS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP1_FUSED_QKV_MM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP2TO6_CREATE_QKV_HEADS_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP8_SOFTMAX_ATTENTION_MASK_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP10_CONCAT_ATTENTION_HEADS_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # MHA SELFOUT ATTENTION
    "OP11_SELFOUT_WEIGHTS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP11_SELFOUT_BIAS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP11_SELFOUT_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # MHA LAYERNORM
    "OP12_LAYERNORM_GAMMA_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP12_LAYERNORM_BETA_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP12_LAYERNORM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # FFN
    "OP13_FF1_MM_WEIGHTS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP13_FF1_MM_BIAS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP13_FF1_MM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP14_FF2_MM_WEIGHTS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP14_FF2_MM_BIAS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "OP14_FF2_MM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # FFN LAYERNORM
    "OP15_LAYERNORM_GAMMA_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP15_LAYERNORM_BETA_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    "OP15_LAYERNORM_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    # After all encoders
    "QA_LINEAR_WEIGHTS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "QA_LINEAR_BIAS_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    "QA_LINEAR_OUTPUT_MEMCFG": ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP1_FUSED_QKV_MM_WEIGHTS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP1_FUSED_QKV_MM_BIAS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP1_FUSED_QKV_MM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP8_SOFTMAX_ATTENTION_MASK_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP9_POST_SOFTMAX_BMM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    # MHA SELFOUT ATTENTION
    "OP11_SELFOUT_WEIGHTS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP11_SELFOUT_BIAS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP11_SELFOUT_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT16,
    # MHA LAYERNORM
    "OP12_LAYERNORM_GAMMA_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP12_LAYERNORM_BETA_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP12_LAYERNORM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT16,  # Used for ffn sub-graph test, might need in the future with mixed precision
    # FFN
    "OP13_FF1_MM_WEIGHTS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP13_FF1_MM_BIAS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP13_FF1_MM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP14_FF2_MM_WEIGHTS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP14_FF2_MM_BIAS_DTYPE": ttl.tensor.DataType.BFLOAT8_B,
    "OP14_FF2_MM_OUTPUT_DTYPE": ttl.tensor.DataType.BFLOAT16,
    # FFN LAYERNORM
    "OP15_LAYERNORM_GAMMA_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "OP15_LAYERNORM_BETA_DTYPE": ttl.tensor.DataType.BFLOAT16,
    # After all encoders
    "QA_LINEAR_WEIGHTS_DTYPE": ttl.tensor.DataType.BFLOAT16,
    "QA_LINEAR_BIAS_DTYPE": ttl.tensor.DataType.BFLOAT16,
}


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.08,
            9.6,
        ),
    ),
)
def test_perf(use_program_cache, expected_inference_time, expected_compile_time):
    model_config = MIXED_PRECISION_BATCH8_MODEL_CONFIG

    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(
        device,
        ttl.device.MemoryAllocator.BASIC
        if model_config["DEFAULT_MEMCFG"].buffer_type == ttl.tensor.BufferType.DRAM
        else ttl.device.MemoryAllocator.L1_BANKING,
    )
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    HF_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    HF_model.eval()
    tt_model = TtBertBatchDram(HF_model.config, HF_model, device, model_config)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]
    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt",
    )
    tt_input = tt_model.model_preprocessing(**inputs)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(first_key, force_enable=True)
        del tt_output
        tt_input = tt_model.model_preprocessing(**inputs)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_model(1, *tt_input)
        ttl.device.Synchronize()
        profiler.end(second_key, force_enable=True)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    ttl.device.CloseDevice(device)

    prep_report("bert15", BATCH_SIZE, first_iter_time, second_iter_time, comments, cpu_time)
    compile_time = first_iter_time - second_iter_time
    logger.info(f"bert15 inference time: {second_iter_time}")
    logger.info(f"bert15 compile time: {compile_time}")
    assert second_iter_time < expected_inference_time, "bert15 is too slow"
    assert compile_time < expected_compile_time, "bert15 compile time is too slow"
