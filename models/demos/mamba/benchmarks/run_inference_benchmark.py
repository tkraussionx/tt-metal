import time
import argparse
from dataclasses import dataclass

import torch

from transformers import AutoTokenizer


@dataclass
class InferenceBenchmarkResult:
    total_time: float
    tokens_per_s: float
    sequence: str


MODEL_VERISON = "state-spaces/mamba-370m"


def run_inference_benchmark(model_type: str, sequence_length: int = 128):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if model_type == "cpu":

        from models.demos.mamba.reference.decode_model import MambaDecode

        model = MambaDecode.from_pretrained(MODEL_VERISON)

        def generate_cpu(input):
            with torch.no_grad():
                next_token_logits = model(input)
                return next_token_logits

        generate = generate_cpu

    elif model_type == "gpu":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(device="cuda")

        def generate_gpu(input):
            with torch.no_grad():
                next_token_logits = model(input)
                return next_token_logits

        generate = generate_gpu

    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    sequence = tokenizer("_", return_tensors="pt").input_ids

    start = time.time()
    for idx in range(sequence_length):
        next_token_logits = generate(sequence[:, idx])
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        sequence = torch.cat([sequence, next_token], dim=1)
    end = time.time()

    return InferenceBenchmarkResult(
        total_time=end - start,
        tokens_per_s=float(sequence_length) / (end - start),
        sequence=[tokenizer.decode(output.tolist()) for output in sequence][0],
    )


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("--model", required=True, choices=["cpu"], help="The mode under test")
    parser.add_argument("--genlen", default=128, type=int, help="Sequence generation length")
    args = parser.parse_args()

    res = run_inference_benchmark(model_type=args.model, sequence_length=args.genlen)
    print(res)


if __name__ == "__main__":
    main()
