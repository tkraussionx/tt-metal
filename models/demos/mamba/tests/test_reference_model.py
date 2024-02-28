import torch
import pytest
from loguru import logger

from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.reference.model import Mamba
from models.demos.mamba.reference.mamba_decode import generate_through_decode, generate_through_selective_scan

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "model_version, batch, genlen",
    (
        ("state-spaces/mamba-130m", 1, 32),
        ("state-spaces/mamba-370m", 1, 32),
    ),
)
def test_cpu_reference_model_decode_vs_selective_scan(
    model_version: MambaPretrainedModelName,
    batch: int,
    genlen: int,
):
    prompt = "Hello World!"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    selective_scan_model = Mamba.from_pretrained(model_version)
    decode_model = MambaDecode.from_pretrained(model_version)

    selective_scan_output = generate_through_selective_scan(selective_scan_model, tokenizer, prompt, genlen)
    decode_output = generate_through_decode(decode_model, tokenizer, prompt, genlen)

    assert selective_scan_output == decode_output, "Model outputs should match"
