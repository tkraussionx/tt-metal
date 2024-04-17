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

    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        # "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "LM_HEAD_OUTPUT",
        "ATTN_W_LAYOUT",
        # RMS norm
        "NORM_W_LAYOUT",
        "NORM_WEIGHTS",
        # MoE
        "GATE_W_LAYOUT",
        "GATE_WEIGHTS",
        "GATE_MM_OUTPUT",
        # Output
        "OUTPUT_W_LAYOUT",
        "OUTPUT_WEIGHTS",
        "OUTPUT_MM",
    )

    def __init__(self, device=None, model_base_path="/proj_sw/user_dev/hf_data/mistral", instruct=False):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        self.instruct = instruct
        if instruct:  # Load instruct weights and tokenizer (Mixtral-8x7B-Instruct-v0.1)
            self.consolidated_weights_path = lambda i: str(
                self.model_base_path / f"Mixtral-8x7B-v0.1/instruct/consolidated_instruct.{i:02d}.pt"
            )
            self.tokenizer_path = str(self.model_base_path / "Mixtral-8x7B-v0.1/instruct/tokenizer_instruct.model")
            self.state_dict_path = str(
                self.model_base_path / "Mixtral-8x7B-v0.1/instruct/partial_state_dict_instruct.pt"
            )
        else:  # Load generative weights and tokenizer (Mixtral-8x7B-v0.1)
            self.consolidated_weights_path = lambda i: str(
                self.model_base_path / f"Mixtral-8x7B-v0.1/consolidated.{i:02d}.pt"
            )
            self.tokenizer_path = str(self.model_base_path / "Mixtral-8x7B-v0.1/tokenizer.model")
            self.state_dict_path = str(self.model_base_path / "Mixtral-8x7B-v0.1/partial_state_dict.pt")

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        self.model_config["Q_TRANSPOSE_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config[
            "K_CACHE_SLICE_OUTPUT_MEMCFG"
        ] = lambda padded_layer_past_len: ttnn.create_sharded_memory_config(
            shape=(128, padded_layer_past_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config[
            "V_CACHE_SLICE_OUTPUT_MEMCFG"
        ] = lambda padded_layer_past_len: ttnn.create_sharded_memory_config(
            shape=(padded_layer_past_len, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config[
            "ATTN_BATCHED_MM_OUTPUT_MEMCFG"
        ] = lambda padded_layer_past_len: ttnn.create_sharded_memory_config(
            shape=(32, padded_layer_past_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        shard_height = 32
        hidden_size = 4096
        shard_width_hidden_dim_across_32_cores = hidden_size // 32
        self.model_config["SHARDED_NORM_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width_hidden_dim_across_32_cores),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.model_config["SHARDED_NORM_OUTPUT_MEMCFG"] = self.model_config["SHARDED_NORM_INPUT_MEMCFG"]
        self.model_config[
            "SHARDED_NORM_PRGM_CFG"
        ] = ttnn.experimental.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=4,
            block_h=shard_height // 32,
            block_w=shard_width_hidden_dim_across_32_cores // 32,
            inplace=False,
        )

        if device is not None:  # Avoid issue with test_mistral_torch.py not having a device
            grid_size = device.compute_with_storage_grid_size()
            # TODO Lower max grid size (used by MLP) to avoid hangs
            self.max_grid_size = ttnn.CoreGrid(y=7, x=6)  # (y,x)  (y=7, x=8)
            # self.max_grid_size = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)  # (y,x)  (y=7, x=8)
            self.core_grid_attention = (
                ttnn.CoreGrid(y=4, x=8) if (4 <= grid_size.y and 8 <= grid_size.x) else self.max_grid_size
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_attn_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_base_path
                / {
                    ttnn.bfloat16: "mixtral_tensor_cache_instruct_bf16",
                    ttnn.bfloat8_b: "mixtral_tensor_cache_instruct_bfp8",
                }[dtype]
            )
        else:
            return (
                self.model_base_path
                / {ttnn.bfloat16: "mixtral_tensor_cache_bf16", ttnn.bfloat8_b: "mixtral_tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_compute_kernel_config(self):
        return self.compute_kernel_config

    def get_compute_kernel_attn_config(self):
        return self.compute_kernel_attn_config
