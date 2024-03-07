# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Tuple
import tt_lib
from tt_lib import fallback_ops
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, torch_to_tt_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear, format_tensor, unpad_from_zero


class TtAttention(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        tt_cache_path=None,
        output_mem_config=None,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.base_address = base_address
        self.output_mem_config = output_mem_config
        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wq.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wq = TtLinear(
            args.dim,
            args.n_heads * args.head_dim,
            self.wq_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wk_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wk.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wk = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wk_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wv_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wv.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wv = TtLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            self.wv_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        self.wo_weights = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + "wo.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.wo = TtLinear(
            args.n_heads * args.head_dim,
            args.dim,
            self.wo_weights,
            device=self.device,
            output_mem_config=self.args.out_mem_config,
        )

        if self.args.FALLBACK_EMPTY:
            self.cache_k = torch.empty(
                args.max_batch_size,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            )
            self.cache_v = torch.empty(args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim)
        else:
            cache_k = tt_lib.tensor.empty(
                [args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim],
                layout=tt_lib.tensor.Layout.ROW_MAJOR,
                device=self.device,
                output_mem_config=self.args.out_mem_config,
            )
            self.cache_k = tt_to_torch_tensor(cache_k).to(torch.float32)
            cache_v = tt_lib.tensor.empty(
                [args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim],
                layout=tt_lib.tensor.Layout.ROW_MAJOR,
                device=self.device,
                output_mem_config=self.args.out_mem_config,
            )
            self.cache_v = tt_to_torch_tensor(cache_v).to(torch.float32)

    def repeat_kv(self, keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tt_lib.tensor.Tensor:
        dim = 2
        keys = torch_to_tt_tensor_rm(keys, self.device)
        values = torch_to_tt_tensor_rm(values, self.device)
        keys = tt_lib.tensor.repeat_interleave(keys, repeats, dim, output_mem_config=self.args.out_mem_config)
        values = tt_lib.tensor.repeat_interleave(values, repeats, dim, output_mem_config=self.args.out_mem_config)
        return keys, values

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        bcast_freq_xq: tt_lib.tensor.complex_tensor,
        bcast_freq_xk: tt_lib.tensor.complex_tensor,
        positions: tt_lib.tensor.Tensor,
        mask: Optional[torch.Tensor],
        seqlen: int,
    ) -> tt_lib.tensor.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            bsz = x.shape()[2]
            attn_mask = attn_masks[i]
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            # Send past KV from host to device
            layer_past = (
                torch2tt_tensor(self.layer_past_list[i][0], self.devices[i]),
                torch2tt_tensor(self.layer_past_list[i][1], self.devices[i]),
            )
            # layer_past = self.layer_past_list[i]
            # TODO layer_past only works for a single device

            # QKV matmuls
            xqkv_fused = tt_lib.operations.primary.matmul_1d(
                x,
                wqkv,
                # program_config=self.model_config["QKV_MM_PROGCFG"],
                # output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            )

            # Reshape and rotary embeddings
            (
                q_heads,  # [seqlen, n_heads, bsz, head_dim]
                k_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
                v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            ) = tt_lib.tensor.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                # output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            xqkv_fused.deallocate()

            q_heads = tt_lib.tensor.rotary_embedding(q_heads, self.tt_cos_cached[i], self.tt_sin_cached[i], start_pos)
            k_heads = tt_lib.tensor.rotary_embedding(k_heads, self.tt_cos_cached[i], self.tt_sin_cached[i], start_pos)

            # KV update
            keys = layer_past[0]  # [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]
            values = layer_past[1]
            tt_lib.tensor.update_cache(layer_past[0], k_heads, current_pos)
            tt_lib.tensor.update_cache(layer_past[1], v_heads, current_pos)
            # tt_lib.tensor.update_cache(keys, k_heads, current_pos)
            # tt_lib.tensor.update_cache(values, v_heads, current_pos)

            k_heads.deallocate()
            v_heads.deallocate()

            keys = tt_lib.tensor.unpad(
                layer_past[0],
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["KEYS_OUTPUT_MEMCFG"],
            )
            values = tt_lib.tensor.unpad(
                layer_past[1],
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["VALUES_OUTPUT_MEMCFG"],
            )

            # Send updated KV back to host
            layer_past = [tt2torch_tensor(lp) for lp in layer_past]

            # Attention
            keys = tt_lib.tensor.transpose(keys, -1, -2)  #  [batch, num_kv_heads, dhead, cache_len + seqlen]

            q_heads = tt_lib.tensor.interleaved_to_sharded(
                q_heads, sharded_mem_config=self.model_config["QHEADS_MEMCFG"]
            )
            # dynamic sharding
            self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                tt_lib.tensor.BufferType.L1,
                tt_lib.tensor.ShardSpec(
                    tt_lib.tensor.CoreRangeSet(
                        {
                            tt_lib.tensor.CoreRange(
                                tt_lib.tensor.CoreCoord(0, 0),
                                tt_lib.tensor.CoreCoord(7, 3),
                            ),
                        }
                    ),
                    [
                        8 * 1 * 128,
                        padded_layer_past_len,
                    ],
                    tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            keys = tt_lib.tensor.interleaved_to_sharded(
                keys, sharded_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"]
            )

            attn = tt_lib.operations.primary.transformers.group_attn_matmul(
                q_heads,
                keys,
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                # output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
            )  # seqlen, n_heads, batch, cache_len + seqlen

            scale = 1 / math.sqrt(self.head_dim)
            attn = tt_lib.tensor.mul_unary(attn, scale)
            attn = tt_lib.tensor.add(attn, attn_mask)
            attn = tt_lib.tensor.softmax(attn)

            attn_output = tt_lib.operations.primary.transformers.group_attn_matmul(
                attn,
                values,
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                # output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
            )  # seqlen, n_heads, batch, dhead

            attn.deallocate()
            keys.deallocate()
            values.deallocate()
            q_heads.deallocate()

            attn_output = tt_lib.tensor.nlp_concat_heads(
                attn_output,
                # output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )  # seqlen, 1, batch, hidden_size

            dense_out = tt_lib.operations.primary.matmul_1d(
                attn_output,
                wo,
                # compute_with_storage_grid_size=device.compute_with_storage_grid_size()
            )  # seqlen, 1, batch, hidden_size

            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return tt_all_reduce(dense_outputs)
        else:
            return dense_outputs
