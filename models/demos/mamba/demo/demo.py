import argparse

import torch

from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaDecode

GENERATED_SEQUENCE_LENGTH = 64


def get_model():
    return MambaDecode.from_pretrained("state-spaces/mamba-370m")


def display_tokens(tokens: list[str]):
    print("\n" * 1000)
    for text in tokens:
        print(f"{text}\n")
        print("-" * 80)  # Print a separator line for readability
        print(f"\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--model", choices=["cpu", "wh"], default="wh", help="The model under test")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = get_model()

    prompts = tokenizer(args.prompts, return_tensors="pt").input_ids

    for idx in range(GENERATED_SEQUENCE_LENGTH + prompts.shape[1]):
        logits = model(prompts[:, idx].unsqueeze(1))
        if idx >= prompts.shape[1] - 1:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            prompts = torch.cat([prompts, next_token], dim=1)
            display_tokens(tokenizer.batch_decode(prompts))


if __name__ == "__main__":
    main()
