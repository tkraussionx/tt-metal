# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from models.demos.mixtral8x7b.tt.mixtral_common_ttnn import (
    precompute_freqs,
)
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.model import Transformer
from models.demos.mixtral8x7b.reference.tokenizer import Tokenizer


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def main():
    n_layers = 32
    iterations = 20

    # Can avoid running reference model to speed up the test (unless measuring PCC)
    run_ref_pt = True

    model_args = TtModelArgs()
    model_args.n_layers = n_layers

    state_dict = {}
    for i in range(1 + (n_layers - 1) // 4):
        state_dict_i = torch.load(model_args.consolidated_weights_path(i), map_location="cpu")
        state_dict.update(state_dict_i)

    partial_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    base_address = "feed_forward."
    for l in range(model_args.n_layers):
        pre = f"layers.{l}."
        partial_state_dict[pre + base_address + "gate.weight"] = partial_state_dict[
            pre + "block_sparse_moe.gate.weight"
        ]
        del partial_state_dict[pre + "block_sparse_moe.gate.weight"]

        w1 = partial_state_dict[pre + "block_sparse_moe.w1"].view(8, 14336, 4096)
        w2 = partial_state_dict[pre + "block_sparse_moe.w2"].view(8, 4096, 14336)
        w3 = partial_state_dict[pre + "block_sparse_moe.w3"].view(8, 14336, 4096)
        for i in range(8):
            partial_state_dict[pre + base_address + f"experts.{i}.w1.weight"] = w1[i]
            partial_state_dict[pre + base_address + f"experts.{i}.w2.weight"] = w2[i]
            partial_state_dict[pre + base_address + f"experts.{i}.w3.weight"] = w3[i]
        partial_state_dict.pop(pre + "block_sparse_moe.w1")
        partial_state_dict.pop(pre + "block_sparse_moe.w2")
        partial_state_dict.pop(pre + "block_sparse_moe.w3")

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # TODO Update the prompt
    # prompts = ["It_was_the_best_of_times_"] * 32
    prompts = [""] * 32
    # Space token -> (U+2581) == "▁"

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(partial_state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations
    if run_ref_pt:
        all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    if run_ref_pt:
        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    # Keep track of generated outputs to print out later
    if run_ref_pt:
        all_outputs_ref = []
        all_logits = []

    # After loading the model weights, wait for an input to start the generation
    # print("Waiting for an input to start...")
    # input()

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        if run_ref_pt:  # Run reference model
            freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
            positions = torch.tensor([start_pos])
            # mask = tt2torch_tensor(attn_mask[0])
            print(f"pt_decode_input = {pt_decode_input.shape}")
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)  # mask)
            all_logits.append(ref_output)

        print(f"encoded_prompts[0] = {len(encoded_prompts[0])}")
        if i in range(len(encoded_prompts[0])):
            if run_ref_pt:
                all_outputs_ref.append(tokenizer.decode([encoded_prompts[0][i]]))

            print("Prefilling...")
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Decode the generated token and save it to print out later
            if run_ref_pt:
                pt_out_tok = torch.argmax(ref_output, dim=-1).squeeze(1)
                pt_decode_input = embd(pt_out_tok).view(batch, seqlen, -1)
                all_outputs_ref.append(tokenizer.decode(pt_out_tok.tolist()[0]))
                torch.save(all_logits, "ref_logits.pt")

        # TODO Space decoding is currently not working as expected
        # TODO print All 32 users
        if run_ref_pt:
            print("[User 0] Ref generation: ", "".join(all_outputs_ref))


if __name__ == "__main__":
    main()
