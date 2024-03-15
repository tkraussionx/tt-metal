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
    is_wormhole_b0,
)


class TtFalconRotaryEmbedding(torch.nn.Module):
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        tt_device,
        dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config
        t = torch.arange(
            self.max_seq_len_cached,
            device=inv_freq.device,
            dtype=inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        layer_name = f"{base_url}.{layer_num}.rotary_embedding"
        overwrite_cos, overwrite_sin = False, False
        cos_exists = (
            tt_cache_path / f"{layer_name}.cos_cached_{self.model_config['COS_CACHED_WEIGHTS_DTYPE'].name}.bin"
        ).exists()
        if cos_exists:
            self.tt_cos_cached = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{layer_name}.cos_cached_{self.model_config['COS_CACHED_WEIGHTS_DTYPE'].name}.bin")
            ).to(tt_device, self.model_config["COS_CACHED_WEIGHTS_MEMCFG"])
            overwrite_cos = (
                tt2torch_tensor(self.tt_cos_cached).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
        if not cos_exists or overwrite_cos:
            self.tt_cos_cached = torch2tt_tensor(
                emb.cos()[None, None, :, :],
                tt_device,
                tt_memory_config=self.model_config["COS_CACHED_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["COS_CACHED_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(
                    tt_cache_path / f"{layer_name}.cos_cached_{self.model_config['COS_CACHED_WEIGHTS_DTYPE'].name}.bin"
                ),
                self.tt_cos_cached.cpu(),
            )
        sin_exists = (
            tt_cache_path / f"{layer_name}.sin_cached_{self.model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}.bin"
        ).exists()
        if sin_exists:
            self.tt_sin_cached = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{layer_name}.sin_cached_{self.model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}.bin")
            ).to(tt_device, self.model_config["SIN_CACHED_WEIGHTS_MEMCFG"])
            overwrite_sin = (
                tt2torch_tensor(self.tt_sin_cached).shape[-2] != self.max_seq_len_cached
            )  # Verify cached tensor has same max seq len
        if not sin_exists or overwrite_sin:
            self.tt_sin_cached = torch2tt_tensor(
                emb.sin()[None, None, :, :],
                tt_device,
                tt_memory_config=self.model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["SIN_CACHED_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(
                    tt_cache_path / f"{layer_name}.sin_cached_{self.model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}.bin"
                ),
                self.tt_sin_cached.cpu(),
            )

    def forward(self, layer: tt_lib.tensor.Tensor, token_idx: Optional[int] = None) -> tt_lib.tensor.Tensor:
        seq_len = layer.get_legacy_shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"

        return tt_lib.tensor.rotary_embedding(
            layer,
            self.tt_cos_cached,
            self.tt_sin_cached,
            token_idx,
            output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
        )


class TtFalconAttention(nn.Module):
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        model_config=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.device = device
        self.state_dict = state_dict
        self.model_config = model_config
        self.padded_local_heads = 96

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        layer_name = f"{base_url}.{layer_num}.self_attention"
        query_key_value_str = f"{layer_name}.query_key_value.weight"
        selfout_str = f"{layer_name}.dense.weight"

        if (
            tt_cache_path / f"{query_key_value_str}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
        ).exists():
            self.query_key_value_weights = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{query_key_value_str}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"])
        else:
            self.query_key_value_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[query_key_value_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(
                    tt_cache_path / f"{query_key_value_str}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
                ),
                self.query_key_value_weights.cpu(),
            )

        if (tt_cache_path / f"{selfout_str}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}.bin").exists():
            self.dense_weights = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{selfout_str}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"])
        else:
            self.dense_weights = torch2tt_tensor(
                torch.transpose(
                    self.state_dict[selfout_str],
                    -2,
                    -1,
                ),
                self.device,
                tt_memory_config=self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["SELFOUT_MM_WEIGHTS_DTYPE"],
            )
            tt_lib.tensor.dump_tensor(
                str(tt_cache_path / f"{selfout_str}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}.bin"),
                self.dense_weights.cpu(),
            )

        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.device,
            self.head_dim,
            base_url,
            layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.scale = 1 / math.sqrt(self.head_dim)
        self.scalar = pad_by_zero(torch.Tensor([self.scale]), self.device)[0]

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[tt_lib.tensor.Tensor]]]:
        """
        Prefill input shape: [batch, 1, seq_len, hidden_size]
        Decode input shape: [seq_len, 1, batch, hidden_size]
        """
        device = hidden_states.device()

        assert not output_attentions

        if llm_mode == "prefill":
            batch = hidden_states.get_legacy_shape()[0]
            q_len = hidden_states.get_legacy_shape()[2]
            assert layer_past is not None
        elif llm_mode == "decode":
            batch = hidden_states.get_legacy_shape()[2]
            q_len = hidden_states.get_legacy_shape()[0]
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        #################
        ### FUSED QKV ###
        #################
        fused_query_key_value = tt_lib.tensor.falcon_fused_qkv_matmul(
            hidden_states,
            self.query_key_value_weights,
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
        )

        ###########
        ### TMs ###
        ###########
        query_layer, key_layer, value_layer = tt_lib.tensor.nlp_create_qkv_heads_falcon7b(
            fused_query_key_value,
            output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_query_key_value.deallocate()

        #########################
        ### ROTARY EMBEDDINGS ###
        #########################
        if llm_mode == "prefill":
            query_layer = self.rotary_embedding(query_layer)
            key_layer = self.rotary_embedding(key_layer)
        elif llm_mode == "decode":
            query_layer = self.rotary_embedding(query_layer, layer_past_len)
            key_layer = self.rotary_embedding(key_layer, layer_past_len)

        ######################
        ### K CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            tt_lib.tensor.fill_cache(layer_past[0], key_layer, user_id)

        elif llm_mode == "decode":
            # Update kv_cache in place
            tt_lib.tensor.update_cache(layer_past[0], key_layer, layer_past_len)
            # key and value layers will have kv_seq_len padded to nearest 32
            key_layer = tt_lib.tensor.unpad(
                layer_past[0],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                output_mem_config=self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"],
            )
            padded_layer_past_len = nearest_32(layer_past_len + 1)

            key_layer = tt_lib.tensor.interleaved_to_sharded(
                key_layer,
                sharded_mem_config=tt_lib.tensor.MemoryConfig(
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
                        [padded_layer_past_len, 64],
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                ),
            )

            # print(f"1.query_layer:{query_layer.get_legacy_shape()}")

            # Pad and transpose Q for batched matmul
            # if self.batched_attn:
            query_layer = tt_lib.tensor.pad(
                query_layer, [1, self.padded_local_heads, 32, self.head_dim], [0, 0, 0, 0], 0.0
            )

            # print(f"2.query_layer:{query_layer.get_legacy_shape()}")

            query_layer = tt_lib.tensor.transpose(
                query_layer,
                -2,
                -3,
            )

            # print(f"3.query_layer:{query_layer.get_legacy_shape()}")

            query_layer = tt_lib.tensor.reshape(
                query_layer,
                32,
                1,
                self.padded_local_heads,
                self.head_dim,  # Batch must be in dim 0 to match K cache
            )

            # tt_lib.device.Synchronize(device)
            # print(f"4.query_layer:{query_layer.get_legacy_shape()}")

            query_layer = tt_lib.tensor.interleaved_to_sharded(
                query_layer, sharded_mem_config=self.model_config["Q_TRANSPOSE_MEMCFG"]
            )

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            -2,
            -1,
            # output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED, tt_lib.tensor.BufferType.L1
            ),
        )
        key_layer.deallocate()

        if llm_mode == "prefill":
            attn_weights = tt_lib.tensor.matmul(
                query_layer,
                key_layer_transposed,
                output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
            )

        elif llm_mode == "decode":
            # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
            if is_wormhole_b0():
                # attn_weights = tt_lib.operations.primary.transformers.attn_matmul(
                #     query_layer,
                #     key_layer_transposed,
                #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                #     output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                #     output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                # )
                attn_prog_config = self.model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
                attn_output_memcfg = tt_lib.tensor.MemoryConfig(
                    tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                    tt_lib.tensor.BufferType.L1,
                    tt_lib.tensor.ShardSpec(
                        tt_lib.tensor.CoreRangeSet(
                            {
                                tt_lib.tensor.CoreRange(
                                    # Volume must match num batches
                                    tt_lib.tensor.CoreCoord(0, 0),
                                    tt_lib.tensor.CoreCoord(7, 3),
                                ),
                            }
                        ),
                        [
                            96,  # padded heads on each core
                            padded_layer_past_len,
                        ],
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                )

                # print(f"attn_prog_config:{attn_prog_config}")
                # print(f"attn_output_memcfg:{attn_output_memcfg} padded_layer_past_len:{padded_layer_past_len}")

                attn_weights = tt_lib.operations.primary.matmul(
                    query_layer,
                    key_layer_transposed,
                    program_config=attn_prog_config,
                    output_mem_config=attn_output_memcfg,
                    output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
                # tt_lib.device.Synchronize(device)
                # print(f"attn_weights1:{attn_weights.get_legacy_shape()}")

            else:
                attn_weights = tt_lib.operations.primary.transformers.group_attn_matmul(
                    query_layer,
                    key_layer_transposed,
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
        query_layer.deallocate()
        key_layer_transposed.deallocate()

        # print(f"self.scale:{self.scale}")

        # attn_weights = tt_lib.tensor.bcast(
        #     attn_weights,
        #     self.scalar,
        #     tt_lib.tensor.BcastOpMath.MUL,
        #     tt_lib.tensor.BcastOpDim.HW,
        #     output_mem_config=self.model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"],
        # )

        # print(f"attention_mask:{attention_mask.get_legacy_shape()}")
        # tt_lib.device.Synchronize(device)

        attn_weights = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_weights,
            self.scale,
            attention_mask,
            program_config=tt_lib.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                subblock_w=1,
                block_h=self.padded_local_heads // 32,
                block_w=padded_layer_past_len // 32,
                math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
                im_data_format=tt_lib.tensor.DataType.BFLOAT16,
            ),
            is_causal_mask=True,
        )

        # tt_lib.device.Synchronize(device)
        # print(f"attn_weights2:{attn_weights.get_legacy_shape()}")

        # if attention_mask is not None:
        #     attn_weights = tt_lib.tensor.add(
        #         attn_weights,
        #         attention_mask,
        #         output_mem_config=self.model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"],
        #     )

        ###############
        ### SOFTMAX ###
        ###############
        # TODO: Replace with scaled_softmax_attention_mask from BERT
        # attn_weights = tt_lib.operations.primary.softmax_in_place(
        #     attn_weights,
        # )

        ######################
        ### V CACHE UPDATE ###
        ######################
        if llm_mode == "prefill":
            tt_lib.tensor.fill_cache(layer_past[1], value_layer, user_id)

        elif llm_mode == "decode":
            # Update kv_cache in place
            tt_lib.tensor.update_cache(layer_past[1], value_layer, layer_past_len)
            value_layer = tt_lib.tensor.unpad(
                layer_past[1],
                [0, 0, 0, 0],
                [batch - 1, 0, nearest_32(layer_past_len + 1) - 1, self.head_dim - 1],
                output_mem_config=self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"],
            )

            kv_cache_memcfg = tt_lib.tensor.MemoryConfig(
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
                        padded_layer_past_len,
                        64,
                    ],
                    tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )

            value_layer = tt_lib.tensor.interleaved_to_sharded(value_layer, sharded_mem_config=kv_cache_memcfg)

            # tt_lib.device.Synchronize(device)
            # print(f"value_layer:{value_layer.get_legacy_shape()}")

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        if llm_mode == "prefill":
            attn_output = tt_lib.tensor.matmul(
                attn_weights,
                value_layer,
                output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
            )

        elif llm_mode == "decode":
            # TODO: switch to group_attn_matmul once multiple q heads is supported (issue #5318)
            if is_wormhole_b0():
                # attn_output = tt_lib.operations.primary.transformers.attn_matmul(
                #     attn_weights,
                #     value_layer,
                #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                #     output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                #     output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                # )

                scores_prog_config = tt_lib.operations.primary.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=[8, 4],
                    in0_block_w=padded_layer_past_len // 32,  # SEQ_LEN // TILE_DIM (dynamic)
                    out_subblock_h=1,  # TODO: Maximize
                    out_subblock_w=1,  # TODO: Maximize
                    per_core_M=96 // 32,  # N_HEADS_PADDED // TILE_SIZE,
                    per_core_N=64 // 32,  # HEAD_DIM // TILE_SIZE (dynamic)
                )

                scores_batched_mm_output_memcfg = tt_lib.tensor.MemoryConfig(
                    tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                    tt_lib.tensor.BufferType.L1,
                    tt_lib.tensor.ShardSpec(
                        tt_lib.tensor.CoreRangeSet(
                            {
                                tt_lib.tensor.CoreRange(
                                    # Volume must match # of attn heads
                                    tt_lib.tensor.CoreCoord(0, 0),
                                    tt_lib.tensor.CoreCoord(7, 3),
                                ),
                            }
                        ),
                        [
                            96,  # heads
                            64,  # head dim
                        ],
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                )

                # tt_lib.device.Synchronize(device)
                # print(f"attn_weights3:{attn_weights.get_legacy_shape()}")

                attn_output = tt_lib.operations.primary.matmul(
                    attn_weights,
                    value_layer,
                    program_config=scores_prog_config,
                    output_mem_config=scores_batched_mm_output_memcfg,
                    output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )

                # tt_lib.device.Synchronize(device)
                # print(f"attn_output1:{attn_output.get_legacy_shape()}")

            else:
                attn_output = tt_lib.operations.primary.transformers.group_attn_matmul(
                    attn_weights,
                    value_layer,
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
        attn_weights.deallocate()
        value_layer.deallocate()

        #########################
        ### ATTENTION SELFOUT ###
        #########################

        attn_output = tt_lib.tensor.sharded_to_interleaved(
            attn_output,
            output_mem_config=self.model_config["L1_MEMCFG"],
        )

        # Get batch in dim 1
        attn_output = tt_lib.tensor.reshape(attn_output, 1, 32, 96, 64)

        # Get batch in dim 2
        attn_output = tt_lib.tensor.transpose(
            attn_output,
            -2,
            -3,
        )

        # UNPAD
        attn_output_shape = attn_output.get_legacy_shape()
        attn_output = tt_lib.tensor.unpad(
            attn_output,
            [0, 0, 0, 0],
            [
                attn_output_shape[0] - 1,
                self.num_heads - 1,
                attn_output_shape[2] - 1,
                attn_output_shape[3] - 1,
            ],
            output_mem_config=self.model_config["L1_MEMCFG"],
        )

        # tt_lib.device.Synchronize(device)
        # print(f"attn_output2:{attn_output.get_legacy_shape()}")

        attn_output = tt_lib.tensor.nlp_concat_heads(
            attn_output,
            output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )

        attn_output = tt_lib.tensor.falcon_selfout_matmul(
            attn_output,
            self.dense_weights,
            output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
        )

        return attn_output, layer_present
