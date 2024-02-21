# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    pad_by_zero,
    nearest_32,
)
from models.experimental.mistral.tt.mistral_common import (
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
    tt_all_reduce,
)


class TtMistralAttention(nn.Module):
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration):
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
        self.sliding_window = configuration.sliding_window

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.model_config = model_config

        self.current = 0

        if layer_num:
            layer_name = f"{base_url}.{layer_num}.attention."
        else:
            layer_name = base_url
        wq_str = f"{layer_name}wq.weight"
        wk_str = f"{layer_name}wk.weight"
        wv_str = f"{layer_name}wv.weight"
        wo_str = f"{layer_name}wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv_list = []
        self.wo_list = []
        self.layer_past_list = []

        for i in range(self.num_devices):
            wqkv = torch2tt_tensor(
                torch.concat(
                    [
                        torch.transpose(
                            torch.chunk(self.state_dict[wq_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wk_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wv_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                    ],
                    dim=-1,
                ),
                self.devices[i],
                tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
            )

            wo = torch2tt_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                self.devices[i],
                tt_memory_config=self.model_config["WO_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["WO_MM_WEIGHTS_DTYPE"],
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
            layer_past = [torch2tt_tensor(lp, self.devices[i]) for lp in layer_past]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)
        self.tt_sin_cached = None
        self.tt_cos_cached = None

    def get_rotation_mat(self, cos, sin, start_pos, seqlen, batch):
        # cos, sin = tt_precompute_freqs(dhead, end)
        rot_mat = freqs_to_rotation_matrix(cos, sin)
        position_ids = torch.ones(batch, seqlen, dtype=torch.long) * start_pos
        rot_emb = tt_gather_rotary_emb(rot_mat, position_ids)
        return rot_emb

    def prepare_inputs(self, x, start_pos, cos, sin):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
        rot_mat = self.get_rotation_mat(cos, sin, start_pos=start_pos, seqlen=seq_len, batch=batch)

        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        self.current = start_pos % self.sliding_window
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)

        if start_pos < self.sliding_window:
            attn_mask[:, :, :, self.current + 1 :] = torch.finfo(attn_mask.dtype).min
        else:
            attn_mask[:, :, :, : self.current] = torch.finfo(attn_mask.dtype).min
            attn_mask[:, :, :, self.sliding_window - self.current :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # TODO: mask out >sliding_window prev tokens

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, bsz, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)

        xs, rot_mats, attn_masks = [], [], []
        for i in range(self.num_devices):
            device = self.devices[i]
            xs.append(torch2tt_tensor(x.clone(), device))
            rot_mats.append(torch2tt_tensor(rot_mat.clone(), device))
            attn_masks.append(torch2tt_tensor(attn_mask.clone(), device))

            self.tt_sin_cached = torch2tt_tensor(sin.clone(), device)
            self.tt_cos_cached = torch2tt_tensor(cos.clone(), device)
            print("COS SHAPE", self.tt_sin_cached.shape(), self.tt_sin_cached.shape())

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        rot_mat: ???
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            bsz = x.shape()[2]
            rot_mat = rot_mats[i]
            attn_mask = attn_masks[i]
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            ###
            # QKV matmuls
            ###
            xqkv_fused = tt_lib.operations.primary.matmul_1d(
                x,
                wqkv,
                # program_config=self.model_config["QKV_MM_PROGCFG"],
                # output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                # output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            )

            print("MATMUL DONE")
            ###
            # Reshape and rotary embeddings
            ###

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

            print("HEADS DONE")

            # Have to put bsz back in dim 1 to match rot_mat shape
            q_heads = tt_lib.tensor.transpose(q_heads, 1, 2)
            k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)

            q_heads = tt_lib.tensor.bmm(
                q_heads,
                rot_mat,  # [seqlen, bsz, n_heads, head_dim]  # [1, bsz, head_dim, head_dim]
                # output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
            )
            k_heads = tt_lib.tensor.bmm(
                k_heads,
                rot_mat,  # [seqlen, bsz, n_kv_heads, head_dim]  # [1, bsz, head_dim, head_dim]
                # output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
            )

            q_heads = tt_lib.tensor.transpose(
                q_heads, 1, 2
            )  # , output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],)
            k_heads = tt_lib.tensor.transpose(k_heads, 1, 2)
            """
            q_heads = tt_lib.tensor.rotary_embedding(
                    q_heads,
                    self.tt_cos_cached,
                    self.tt_sin_cached,
                    start_pos)

            k_heads = tt_lib.tensor.rotary_embedding(
                    k_heads,
                    self.tt_cos_cached,
                    self.tt_sin_cached,
                    start_pos)
            """
            print("ROT DONE")
            ###
            # KV update
            ###

            keys = layer_past[0]
            values = layer_past[1]
            # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
            # v_heads [seqlen, n_kv_heads, bsz, head_dim]
            # keys, [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]

            tt_lib.tensor.update_cache(keys, k_heads, self.current)
            tt_lib.tensor.update_cache(values, v_heads, self.current)

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
                # output_mem_config=self.model_config["KEYS_MEMCFG"],
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
                # output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )

            print("KV UPDATE DONE")
            ###
            # Attention
            ###

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
            # print("scale2", attn.shape(), attn_mask.shape(), attn, attn_mask)
            attn = tt_lib.tensor.add(attn, attn_mask)
            attn = tt_lib.tensor.softmax(attn)

            print("GQA2:", attn.shape(), values.shape())

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

            print("DENSE SHAPES", attn_output.shape(), wo.shape())
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
