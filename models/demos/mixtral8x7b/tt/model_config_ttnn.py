# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from pathlib import Path


class TtModelArgs:
    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    sliding_window = 4096
    vocab_size = 32000

    max_batch_size = 32
    max_seq_len = 4096
    moe = True
    num_experts = 8
    num_experts_per_tok = 2

    def __init__(self, model_base_path="/proj_sw/user_dev/hf_data/mistral"):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        self.consolidated_weights_path = lambda i: str(
            self.model_base_path / f"Mixtral-8x7B-v0.1/consolidated.{i:02d}.pt"
        )
        self.tokenizer_path = str(self.model_base_path / "Mixtral-8x7B-v0.1/tokenizer.model")

    def weight_cache_path(self, dtype):
        return (
            self.model_base_path
            / {ttnn.bfloat16: "mixtral_tensor_cache_bf16", ttnn.bfloat8_b: "mixtral_tensor_cache_bfp8"}[dtype]
        )
