import argparse
from typing import List

import torch

from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model(version: MambaPretrainedModelName):
    from models.demos.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(version)


def get_tt_metal_model(version: MambaPretrainedModelName):
    import tt_lib
    from models.demos.mamba.tt.full_model import MambaTT

    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    device = tt_lib.device.GetDefaultDevice()
    reference_model = get_cpu_reference_model(version)
    model = MambaTT(reference_model, 48, device)
    return model, device


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def display_tokens(tokens: List[str]):
    print("\n" * 1000)
    for text in tokens:
        print(f"{text}\n")
        print("-" * 80)  # Print a separator line for readability
        print(f"\n")


def run_demo(
    prompts: List[str],
    model_type: str,
    model_version: MambaPretrainedModelName = "state-spaces/mamba-130m",
    generated_sequence_length: int = 32,
    display: bool = True,
):
    if model_type == "cpu":
        model = get_cpu_reference_model(model_version)
    elif model_type == "wh":
        model, _ = get_tt_metal_model(model_version)
    else:
        raise RuntimeError("Invalid model type was encountered")

    tokenizer = get_tokenizer()

    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).input_ids

    all_decoded_sequences = []
    for idx in range(generated_sequence_length + sequences.shape[1]):
        logits = model(sequences[:, idx].unsqueeze(1))
        if idx >= sequences.shape[1] - 1:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            sequences = torch.cat([sequences, next_token], dim=1)

            decoded = tokenizer.batch_decode(sequences)
            all_decoded_sequences.append(decoded)

            if display:
                display_tokens(decoded)

    return all_decoded_sequences


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--model", choices=["cpu", "wh"], default="wh", help="The model under test")
    args = parser.parse_args()
    run_demo(args.prompts, args.model)


if __name__ == "__main__":
    main()
