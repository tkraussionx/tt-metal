# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.llama3.tt.llama_attention import TtLlamaAttention
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule


class TtTransformerBlock(LightweightModule):
    def __init__(self, args, mesh_device, dtype, state_dict, layer_num, weight_cache_path):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = 1

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.sliding_window = args.sliding_window
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.attention = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
        )
        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="attention_norm",
            model_config=self.model_config,
        )
        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="ffn_norm",
            model_config=self.model_config,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
    ) -> ttnn.Tensor:
        print(f"mode: {mode}")
        print(x.shape)

        if mode == "prefill":
            skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
            all_gather_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        else:
            skip_mem_cfg = self.model_config["DEC_SKIP_OUTPUT_MEMCFG"]
            all_gather_mem_cfg = self.model_config["SHARDED_NORM_INPUT_MEMCFG"]

        if self.model_config["IS_MULTICHIP"] and not self.model_config["IS_DISTRIBUTED_NORM"](mode):
            x_gathered = ttnn.all_gather(
                x, dim=3, num_links=1, topology=self.model_config["CCL_TOPOLOGY"], memory_config=all_gather_mem_cfg
            )
            # ttnn.deallocate(x)
        else:
            x_gathered = x

        print(x_gathered.shape)

        attn_norm = self.attention_norm(x_gathered, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))
        # ttnn.deallocate(x_gathered)

        if self.model_config["IS_DISTRIBUTED_NORM"](mode):
            attn_norm_gathered = ttnn.all_gather(
                attn_norm, dim=3, num_links=1, topology=self.model_config["CCL_TOPOLOGY"]
            )
            ttnn.deallocate(attn_norm)
        else:
            attn_norm_gathered = attn_norm

        # Attention module expects a list of inputs (multi-device support)
        r = self.attention.forward(
            attn_norm_gathered,
            current_pos,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
            page_table,
        )

        h = ttnn.add(x, r, memory_config=skip_mem_cfg)
        ttnn.deallocate(x)

        if self.model_config["IS_MULTICHIP"] and not self.model_config["IS_DISTRIBUTED_NORM"](mode):
            h_gathered = ttnn.all_gather(h, dim=3, num_links=1, topology=self.model_config["CCL_TOPOLOGY"])
            ttnn.deallocate(h)
        else:
            h_gathered = h

        ff_normalized = self.ffn_norm(h_gathered)
        # ttnn.deallocate(h_gathered)

        if self.model_config["IS_DISTRIBUTED_NORM"](mode):
            ff_normalized_gathered = ttnn.all_gather(
                ff_normalized, dim=3, num_links=1, topology=self.model_config["CCL_TOPOLOGY"]
            )
            ttnn.deallocate(ff_normalized)
        else:
            ff_normalized_gathered = ff_normalized

        ff_normalized_gathered = self.feed_forward.forward(ff_normalized_gathered, mode)
        out = ttnn.add(h, ff_normalized_gathered, memory_config=skip_mem_cfg)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_normalized_gathered)
        return out
