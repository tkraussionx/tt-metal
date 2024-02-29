import time
import argparse
from dataclasses import dataclass
from typing import List

import torch

from transformers import AutoTokenizer

MODEL_VERSION = "state-spaces/mamba-370m"


@dataclass
class InferenceBenchmarkResult:
    total_time_ms: float
    tokens_per_s: float
    sequence: str


class MambaDecodeWrapper(torch.nn.Module):
    def __init__(self, model_version):
        super().__init__()

        from models.demos.mamba.reference.decode_model import MambaDecode

        self.decode = MambaDecode.from_pretrained(model_version)

    def forward(self, x):
        return self.decode(x)


class MambaGPUWrapper(torch.nn.Module):
    def __init__(self, model_version):
        super().__init__()

        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.decode = MambaLMHeadModel.from_pretrained(model_version, device="cuda")

    def forward(self, x):
        return self.decode(x).logits


def create_model(model_type: str):
    if model_type == "cpu":
        return MambaDecodeWrapper(MODEL_VERSION), "cpu"
    elif model_type == "gpu":
        return MambaGPUWrapper(MODEL_VERSION), "cuda"
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")


def run_inference_benchmark(model_type: str, prompt: str = "Mamba is the", sequence_length: int = 64):
    torch.random.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model, device = create_model(model_type)

    sequence = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device).split(1, dim=1)
    tokens_in_prompt = len(sequence)

    @torch.inference_mode()
    def decode(prompt: List[torch.Tensor]) -> List[torch.Tensor]:
        result = [*prompt]
        for idx in range(sequence_length + len(prompt)):
            logits = model(result[idx])
            if idx >= len(prompt) - 1:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                result.append(next_token)
        return result

    print("Warming up...")
    out = decode(sequence)
    print("Done warming up")

    start = time.time()
    decode(sequence)
    end = time.time()

    return InferenceBenchmarkResult(
        total_time_ms=1000.0 * (end - start),
        tokens_per_s=float(tokens_in_prompt + sequence_length) / (end - start),
        sequence=tokenizer.batch_decode(torch.cat(out, dim=1)),
    )


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("--model", required=True, choices=["cpu", "gpu"], help="The model under test")
    parser.add_argument("--genlen", default=64, type=int, help="Sequence generation length")
    args = parser.parse_args()

    res = run_inference_benchmark(model_type=args.model, sequence_length=args.genlen)
    print(res)


if __name__ == "__main__":
    main()
