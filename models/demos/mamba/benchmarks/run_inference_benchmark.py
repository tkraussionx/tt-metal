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


MODEL_VERSION = "state-spaces/mamba-370m"


def create_model(model_type: str):
    if model_type == "cpu":

        from models.demos.mamba.reference.decode_model import MambaDecode

        model = MambaDecode.from_pretrained(MODEL_VERSION)
        model.eval()

        def generate_cpu(inputs):
            with torch.no_grad():
                next_token_logits = model(inputs)
                return next_token_logits

        return generate_cpu

    elif model_type == "gpu":

        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(MODEL_VERSION, device="cuda")
        model.eval()

        def generate_gpu(inputs):
            with torch.no_grad():
                inputs = inputs.unsqueeze(1).to(device="cuda")
                next_token_logits = model(inputs).logits
                return next_token_logits.squeeze(1)

        return generate_gpu

    else:
        raise RuntimeError(f"Invalid model type: {model_type}")


def run_inference_benchmark(model_type: str, prompt: str = "Mamba is the", sequence_length: int = 64):

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = create_model(model_type)

    sequence = tokenizer(prompt, return_tensors="pt").input_ids
    tokens_in_prompt = len(sequence[0])

    for idx in range(tokens_in_prompt - 1):
        model(sequence[:, idx])

    start = time.time()
    for idx in range(tokens_in_prompt - 1, tokens_in_prompt + sequence_length):
        logits = model(sequence[:, idx])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        sequence = torch.cat([sequence, next_token.to(device="cpu")], dim=1)
    end = time.time()

    return InferenceBenchmarkResult(
        total_time=end - start,
        tokens_per_s=float(sequence_length) / (end - start),
        sequence=[tokenizer.decode(output.tolist()) for output in sequence][0],
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
