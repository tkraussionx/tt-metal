# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
from pathlib import Path
from models.utility_functions import is_wormhole_b0
from loguru import logger
import tarfile
import urllib.request


class TtModelArgs:
    """Model args for Llama 3.1 8B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    vocab_size = 128256

    # Llama 3.1 8B names
    hidden_size = dim
    intermediate_size = hidden_dim
    num_attention_heads = n_heads
    num_hidden_layers = n_layers
    hidden_act = "silu"
    mlp_bias = False
    pretraining_tp = 1

    #   "rope_scaling": {
    #     "factor": 8.0,
    #     "low_freq_factor": 1.0,
    #     "high_freq_factor": 4.0,
    #     "original_max_position_embeddings": 8192,
    #     "rope_type": "llama3"
    #   },
    #   "rope_theta": 500000.0,

    # Parameters for our use
    max_batch_size = 32
    max_seq_len = 131072
    kv_seq_len = 1024  # TODO Update the initial cache size when scaling up (Should be window_size == 4096)

    # Default folder location for weights and cached files
    DEFAULT_CKPT_DIR = os.getenv("LLAMA31_8B_CKPT_DIR", "/proj_sw/user_dev/hf_data/llama/llama31_8b/")
    DEFAULT_CACHE_PATH = os.getenv("LLAMA31_8B_CACHE_PATH", "/proj_sw/user_dev/hf_data/llama/llama31_8b/")

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
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "LM_HEAD_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DEC_SKIP_OUTPUT",
    )

    def __init__(self, device, instruct=False):
        # Assert if all folders and files exist
        assert os.path.exists(
            self.DEFAULT_CKPT_DIR
        ), f"Checkpoint directory {self.DEFAULT_CKPT_DIR} does not exist, please use export LLAMA31_8B_CKPT_DIR=..."
        assert os.path.exists(
            self.DEFAULT_CACHE_PATH
        ), f"Cache directory {self.DEFAULT_CACHE_PATH} does not exist, please use export LLAMA31_8B_CACHE_PATH=..."
        # Check if weights exist in the specified folder. If not warn the user to run the download and untar script.
        assert os.path.isfile(
            self.DEFAULT_CKPT_DIR + "/model.safetensors.index.json"
        ), f"file model.safetensors.index.json file does not exist. Please download the model weights from huggingface."

        logger.info(f"Checkpoint directory: {self.DEFAULT_CKPT_DIR}")
        logger.info(f"Cache directory: {self.DEFAULT_CACHE_PATH}")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.DEFAULT_CKPT_DIR)
        self.model_cache_path = Path(self.DEFAULT_CACHE_PATH)

        # Load weights and tokenizer
        self.weights_index_path = (
            self.DEFAULT_CKPT_DIR + "/model-00001-of-00004.safetensors"
        )  # "/model.safetensors.index.json"

        self.instruct = instruct

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        if device is not None:  # Avoid issue with test_llama_torch.py not having a device
            grid_size = device.compute_with_storage_grid_size()
            for i in range(grid_size.y, 0, -1):
                # Force the number of rows in the grid to be a factor of max_batch_size for a valid sharding
                if self.max_batch_size % i == 0:
                    grid_size_y = i
                    break
            assert (
                self.max_batch_size % grid_size_y == 0
            ), f"Number of rows in the grid should be a factor of max_batch_size ({self.max_batch_size})"
            self.max_grid_size = ttnn.CoreGrid(y=grid_size_y, x=grid_size.x)  # (y,x)

            # Add sharded memory config for MLP FF1/FF3
            mlp_shard_config = ttnn.create_sharded_memory_config(
                [self.max_batch_size, self.hidden_dim], self.max_grid_size, ttnn.ShardStrategy.WIDTH
            )
            self.model_config["FF1_OUTPUT_MEMCFG"] = mlp_shard_config
            self.model_config["FF3_OUTPUT_MEMCFG"] = mlp_shard_config

            # Compute kernel shared by attention and MLP. FP32 acc is needed for accuracy
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_cache_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    def get_compute_kernel_config(self):
        return self.compute_kernel_config
