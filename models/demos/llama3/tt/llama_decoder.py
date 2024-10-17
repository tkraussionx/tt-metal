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
            is_distributed=self.args.is_distributed_norm,
        )
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="ffn_norm",
            model_config=self.model_config,
            is_distributed=self.args.is_distributed_norm,
        )

    def decoder_norm(self, x, norm, mode):
        """Apply a norm, possibly gathering inputs if required."""
        mem_cfg = self.model_config["SHARDED_NORM_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology, memory_config=mem_cfg)
        elif mode == "decode":
            # Gathered norms will be sharded for decode mode, so single-chip should be too
            x = ttnn.interleaved_to_sharded(x, mem_cfg)

        # x sharded in decode mode here
        x = norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm already performs a gather
        if self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology)

        return x

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
        # print(f"mode: {mode}")
        # print(x.shape)

        # Use L1 interleaved for decode because self.decoder_norm's gather requires interleaved inputs
        # FIXME: move to sharded residuals once support for this is added
        skip_mem_cfg = self.model_config["DEC_SKIP_OUTPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Attention Norm
        attn_in = self.decoder_norm(x, self.attention_norm, mode)

        # Attention module expects a list of inputs (multi-device support)
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
            page_table,
        )

        # Residual Add
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)
        ttnn.deallocate(attn_out)

        # FF Norm
        ff_in = self.decoder_norm(h, self.ff_norm, mode)
        ff_out = self.feed_forward.forward(ff_in, mode)

        # Residual Add
        out = ttnn.add(h, ff_out, memory_config=skip_mem_cfg)

        return out
