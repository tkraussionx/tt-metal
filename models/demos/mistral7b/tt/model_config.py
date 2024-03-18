# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from pathlib import Path
from models.utility_functions import is_wormhole_b0


class TtModelArgs:
    """Model args for Mistral 7B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    sliding_window = 4096
    vocab_size = 32000

    # Parameters for our use
    max_batch_size = 32
    max_seq_len = 4096

    max_grid_size = ttnn.CoreGrid(x=8, y=8) if is_wormhole_b0() else ttnn.CoreGrid(x=12, y=9)

    def __init__(self, model_base_path="/proj_sw/user_dev/hf_data/mistral", instruct=False):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        if instruct:  # Load instruct weights and tokenizer (Mistral-7B-Instruct-v0.2)
            self.consolidated_weights_path = str(self.model_base_path / "mistral-7B-v0.1/consolidated_instruct.00.pth")
            self.tokenizer_path = str(self.model_base_path / "tokenizer_instruct.model")
        else:  # Load generative weights and tokenizer (Mistral-7B-v0.1)
            self.consolidated_weights_path = str(self.model_base_path / "mistral-7B-v0.1/consolidated.00.pth")
            self.tokenizer_path = str(self.model_base_path / "mistral-7B-v0.1/tokenizer.model")

    def weight_cache_path(self, dtype, instruct=False):
        # Keep the weight cache separate for generative and instruct weights
        if instruct:
            return (
                self.model_base_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_base_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )
