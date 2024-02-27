import time
import argparse
from dataclasses import dataclass

import torch

from transformers import AutoTokenizer

MODEL_VERSION = "state-spaces/mamba-370m"


@dataclass
class InferenceBenchmarkResult:
    total_time: float
    tokens_per_s: float
    sequence: str


class MambaDecodeWrapper(torch.nn.Module):

    @dataclass
    class Result:
        logits: torch.Tensor

    def __init__(self, model_version):
        super().__init__()

        from models.demos.mamba.reference.decode_model import MambaDecode

        self.decode = MambaDecode.from_pretrained(model_version)

    def forward(self, inputs):
        x = inputs.squeeze(0)
        return MambaDecodeWrapper.Result(self.decode(x).unsqueeze(1))


def create_model(model_type: str):
    if model_type == "cpu":
        model = MambaDecodeWrapper(MODEL_VERSION)
        device = "cpu"
    elif model_type == "gpu":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(MODEL_VERSION, device="cuda")
        device = "cuda"
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    return model, device


def run_inference_benchmark(model_type: str, prompt: str = "Mamba is the", sequence_length: int = 64):

    torch.random.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model, device = create_model(model_type)

    sequence = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device).split(1, dim=1)
    tokens_in_prompt = len(sequence)

    @torch.inference_mode()
    def decode(prompt: list[torch.Tensor]) -> list[torch.Tensor]:
        for idx in range(len(prompt) - 1):
            model(prompt[idx])

        sequence = [*prompt]
        for idx in range(sequence_length):
            logits = model(sequence[-1]).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            sequence.append(next_token)
        return sequence

    print("Warming up...")
    decode(sequence)
    print("Done warming up")

    start = time.process_time()
    result = decode(sequence)
    end = time.process_time()

    result = torch.cat(result, dim=1)
    assert result.shape[1] == tokens_in_prompt + sequence_length

    return InferenceBenchmarkResult(
        total_time=1000.0 * (end - start),
        tokens_per_s=float(tokens_in_prompt + sequence_length) / (end - start),
        sequence=tokenizer.batch_decode(result),
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
