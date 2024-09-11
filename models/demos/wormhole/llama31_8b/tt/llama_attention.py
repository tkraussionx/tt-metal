# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from torch import nn

import ttnn
from models.utility_functions import (
    nearest_32,
)


class TtLlamaAttention(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.kv_seq_len = configuration.kv_seq_len
        self.sliding_window = configuration.sliding_window
        self.grid_size = configuration.max_grid_size

        self.model_config = configuration.get_model_config()
        self.compute_kernel_config = configuration.get_compute_kernel_config()

        layer_name = f"layers.{layer_num}.attention"
        cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
                    torch.transpose(
                        torch.chunk(self.state_dict[wq_str], self.num_devices)[0],
                        -2,
                        -1,
                    ),
                    torch.transpose(
                        torch.chunk(self.state_dict[wk_str], self.num_devices)[0],
                        -2,
                        -1,
                    ),
                    torch.transpose(
                        torch.chunk(self.state_dict[wv_str], self.num_devices)[0],
                        -2,
                        -1,
                    ),
                ],
                dim=-1,
            ),
            device=self.devices[0],
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name("wqkv"),
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[0],
                -2,
                -1,
            ),
            device=self.devices[0],
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            dtype=self.dtype,
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name("wo"),
        )

        cache_k = torch.zeros(
            (
                self.max_batch_size,
                self.n_kv_heads // self.num_devices,
                self.sliding_window,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.n_kv_heads // self.num_devices,
                self.sliding_window,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.from_torch(
                lp, device=self.devices[0], layout=self.model_config["ATTN_W_LAYOUT_TILE"], dtype=self.dtype
            )
            for lp in layer_past
        ]

        self.q_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.k_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.expand_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=4,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.reduce_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.attn_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
        )
        self.compute_kernel_config_attn = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_grid = ttnn.CoreCoord(8, 4)
        self.scale = self.head_dim**-0.5

    def forward_decode(
        self,
        xs,
        current_pos,
        rot_mat=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        """
        x = xs
        wqkv = self.wqkv
        wo = self.wo
        layer_past = self.layer_past
        assert self.max_batch_size * self.n_kv_heads < 64
        ###
        # QKV matmuls
        ###
        xqkv_fused = ttnn.linear(
            x,
            wqkv,
            memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            core_grid=self.grid_size,
        )

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, ttnn.Shape((1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
        )

        # ttnn.deallocate(x)

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot,  # [seqlen, n_heads, bsz, head_dim]
            k_heads_pre_rot,  # [seqlen, n_kv_heads, bsz, head_dim]
            v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=self.model_config["QKV_HEADS_OUTPUT_MEMCFG"],
        )

        ttnn.deallocate(xqkv_fused)

        # Update rotary matrix on device
        rotary_mat = rot_mat

        q_heads = ttnn.linear(
            q_heads_pre_rot,
            rotary_mat,
            # program_config=self.q_heads_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        k_heads = ttnn.linear(
            k_heads_pre_rot,
            rotary_mat,
            # program_config=self.k_heads_program_config,
            memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
        )

        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        ###
        # KV update
        ###
        keys = layer_past[0]
        values = layer_past[1]

        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]
        ttnn.kv_cache.update_cache_for_token_(keys, k_heads, 0)
        ttnn.kv_cache.update_cache_for_token_(values, v_heads, 0)
        self.layer_past = [keys, values]

        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode_gqa(
            q_heads,
            keys,
            values,
            [0] * self.max_batch_size * self.n_kv_heads,  # FIXME current_pos,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D, memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"]
        )
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_heads,
        )
        attn_output_cat = ttnn.reshape(attn_output_cat, ttnn.Shape((1, 1, 32, self.hidden_size)))

        dense_out = ttnn.linear(
            attn_output_cat,
            wo,
            memory_config=self.model_config["ATTN_OUTPUT_MEMCFG"],
            program_config=self.model_config["ATTN_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel_config,
            # core_grid=self.grid_size,
        )  # seqlen, 1, batch, hidden_size

        ttnn.deallocate(attn_output_cat)

        return dense_out

    def forward_prefill(self, xs_11SH, rot_mats, transformation_mats, user_id: int = 0):
        seq_len = xs_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        wqkv = self.wqkv
        wo = self.wo

        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            xs_11SH = ttnn.reshape(xs_11SH, [1, 2, seq_len // 2, -1])
        xqkv_fused = ttnn.linear(
            xs_11SH,
            wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        xs_11SH.deallocate(True)

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        q_heads_1QSD_pre_rot.deallocate(True)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        k_heads_1KSD_pre_rot.deallocate(True)

        # Fill KV-Cache
        keys_BKSD = self.layer_past[0]
        values_BKSD = self.layer_past[1]

        k_fill = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        # sharding k_fill to deal with update_cache memory limitation
        if seq_len > 128:
            k_fill = ttnn.interleaved_to_sharded(k_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        v_fill = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
        # sharding v_fill to deal with update_cache memory limitation
        if seq_len > 128:
            v_fill = ttnn.interleaved_to_sharded(v_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        ttnn.fill_cache(
            keys_BKSD,
            k_fill,
            user_id,
        )
        ttnn.fill_cache(
            values_BKSD,
            v_fill,
            user_id,
        )

        self.layer_past = [keys_BKSD, values_BKSD]

        # SDPA

        # reshaping to put group in batch dim to do sdpa on 8 MQAs in parallel
        k_heads_K1SD = ttnn.reshape(k_heads_1KSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        v_heads_V1SD = ttnn.reshape(v_heads_1VSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        q_heads_84SD = ttnn.reshape(
            q_heads_1QSD, [self.n_local_kv_heads, self.n_local_heads // self.n_local_kv_heads, -1, self.head_dim]
        )
        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_84SD,
            k_heads_K1SD,
            v_heads_V1SD,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        # deallocate keys and values
        q_heads_84SD.deallocate(True)
        k_heads_K1SD.deallocate(True)
        v_heads_V1SD.deallocate(True)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_1QSD.deallocate(True)

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 2, seq_len // 2, -1])
        output_11SH = ttnn.linear(
            attn_output_11SH,
            wo,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        attn_output_11SH.deallocate(True)
        return output_11SH

    def forward(self, xs, current_pos, rot_mats=None, transformation_mats=None, user_id=0, mode="decode"):
        if mode == "prefill":
            return self.forward_prefill(xs, rot_mats, transformation_mats, user_id)
        else:
            return self.forward_decode(xs, current_pos, rot_mats)
