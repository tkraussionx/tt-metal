import torch
import pytest
from loguru import logger

from transformers import AutoTokenizer

from models.demos.mamba.demo.demo import run_demo
from models.demos.mamba.reference.decode_model import MambaPretrainedModelName


@pytest.mark.parametrize(
    "model_version, genlen",
    (("state-spaces/mamba-370m", 4),),
)
def test_cpu_reference_model_decode_vs_selective_scan(
    model_version: MambaPretrainedModelName,
    genlen: int,
):
    prompt = ["Hello"]

    res1 = run_demo(prompt, "cpu", generated_sequence_length=genlen, model_version=model_version, display=False)
    assert len(res1) == genlen + 1

    res2 = run_demo(prompt, "wh", generated_sequence_length=genlen, model_version=model_version, display=False)
    assert len(res2) == genlen + 1

    assert res1 == res2, "Model outputs should match"
