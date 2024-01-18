# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from torch.nn import functional as F
import ttnn
from transformers import (
    AutoFeatureExtractor,
    WhisperModel,
    WhisperConfig,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from models.generation_utils import get_logits_processor
from ttnn.model_preprocessing import preprocess_model_parameters


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_generate(
    config,
    input_embeds,
    input_features,
    ttnn_model,
    decoder_hidden_states,
    decoder_attention_mask,
    parameters,
    processor,
    ttnn_linear_weight,
    max_tokens,
    device,
):
    tt_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id

    logits_processor = get_logits_processor(tt_input_ids, config)

    tt_input_ids = pad_input_32(tt_input_ids, config.pad_token_id).to(torch.long)

    for i in range(max_tokens):
        tt_output = ttnn_model.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
        )
        tt_output = tt_output @ ttnn_linear_weight

        tt_output = ttnn.from_device(tt_output)

        logits_to_torch = ttnn.to_torch(tt_output)

        next_token_logits = logits_to_torch[:, i, :]

        next_tokens_scores = logits_processor(input_features, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        tt_input_ids[:, i + 1] = next_tokens[:, None]

        if next_tokens == config.eos_token_id:
            break
        logger.info(processor.batch_decode(tt_input_ids, skip_special_tokens=True)[0])

    logger.info(f"Final transcriptions")
    tt_transcription = processor.batch_decode(tt_input_ids, skip_special_tokens=True)[0]
    logger.info(f"TT transcription: {tt_transcription}")

    return tt_transcription


@pytest.mark.parametrize(
    ("model_name", "max_tokens", "use_optimized_version"),
    (("openai/whisper-base", 5, False),),
)
def test_demo_functional_whisper(model_name, max_tokens, use_optimized_version, device):
    torch.manual_seed(0)

    model = WhisperModel.from_pretrained(model_name).to(torch.bfloat16).eval()

    config = WhisperConfig.from_pretrained(model_name)

    processor = AutoProcessor.from_pretrained(model_name, language="English", task="transcribe")

    whisper_conditional_generation = (
        WhisperForConditionalGeneration.from_pretrained(model_name).to(torch.bfloat16).eval()
    )
    linear_weight = whisper_conditional_generation.proj_out.weight

    linear_weight = F.pad(linear_weight, (0, 0, 0, 1))
    ttnn_linear_weight = ttnn.from_torch(linear_weight)
    ttnn_linear_weight = ttnn.to_device(ttnn_linear_weight, device)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))

    tt_model_name = "ttnn_" + ("optimized_" if use_optimized_version else "") + model_name.replace("/", "_")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    dtype_to_use = torch.bfloat16
    input_features = inputs.input_features.type(dtype_to_use)

    decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
    decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)

    num_heads = config.encoder_attention_heads
    attention_mask = None

    if use_optimized_version:
        ttnn_model = ttnn_optimized_functional_whisper
    else:
        ttnn_model = ttnn_functional_whisper

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        parameters=parameters,
        device=device,
    )

    tt_output = run_generate(
        config,
        input_embeds,
        input_features,
        ttnn_model,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
        processor=processor,
        ttnn_linear_weight=ttnn_linear_weight,
        max_tokens=max_tokens,
        device=device,
    )
    logger.info("tt_output")
    logger.info(tt_output)
