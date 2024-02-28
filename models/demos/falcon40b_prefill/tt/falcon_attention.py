# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon40b_prefill.tt.model_utils import memcfg_for_seqlen

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    pad_by_zero,
    nearest_32,
)


def generate_cos_sin_cache(
    tt_devices,
    head_dim,
    base_url,
    max_position_embeddings=2048,
    base=10000,
    model_config=None,
    tt_cache_path=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)

    layer_name = f"{base_url}.rotary_embedding_base_{base}_head_dim_{head_dim}_seq_len_{max_position_embeddings}"
    cos_cached_path = tt_cache_path / f"{layer_name}.cos_cached_{model_config['COS_CACHED_WEIGHTS_DTYPE'].name}.bin"
    if (cos_cached_path).exists():
        tt_cos_cached_host = tt_lib.tensor.load_tensor(str(cos_cached_path))
        tt_cos_cached = [
            tt_cos_cached_host.to(tt_device, model_config["COS_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
        ]
    else:
        tt_cos_cached_host = torch2tt_tensor(
            emb.cos()[None, None, :, :],
            None,
            tt_memory_config=model_config["COS_CACHED_WEIGHTS_MEMCFG"],
            tt_dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
        )
        tt_cos_cached = [
            tt_cos_cached_host.to(tt_device, model_config["COS_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
        ]
        tt_lib.tensor.dump_tensor(
            str(cos_cached_path),
            tt_cos_cached_host,
        )
    sin_cached_path = tt_cache_path / f"{layer_name}.sin_cached_{model_config['SIN_CACHED_WEIGHTS_DTYPE'].name}.bin"
    if (sin_cached_path).exists():
        tt_sin_cached_host = tt_lib.tensor.load_tensor(str(sin_cached_path))
        tt_sin_cached = [
            tt_sin_cached_host.to(tt_device, model_config["SIN_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
        ]
    else:
        tt_sin_cached_host = torch2tt_tensor(
            emb.sin()[None, None, :, :],
            None,
            tt_memory_config=model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
            tt_dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
        )
        tt_sin_cached = [
            tt_sin_cached_host.to(tt_device, model_config["SIN_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
        ]
        tt_lib.tensor.dump_tensor(
            str(sin_cached_path),
            tt_sin_cached_host,
        )
    return tt_cos_cached, tt_sin_cached


class TtFalconRotaryEmbedding:
    """
    See FalconRotaryEmbedding from hf_modeling_falcon.py
    """

    def __init__(
        self,
        tt_devices,
        head_dim,
        base_url,
        layer_num,
        max_position_embeddings=2048,
        base=10000,
        model_config=None,
        tt_cache_path=None,
        global_cos_sin_cache=None,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.model_config = model_config
        if global_cos_sin_cache is not None:
            self.tt_cos_cached, self.tt_sin_cached = global_cos_sin_cache
        else:
            self.tt_cos_cached, self.tt_sin_cached = generate_cos_sin_cache(
                tt_devices,
                head_dim,
                f"{base_url}.{layer_num}",
                max_position_embeddings,
                base,
                model_config,
                tt_cache_path,
            )

    def __call__(self, layer: tt_lib.tensor.Tensor, token_idx: Optional[int] = None) -> tt_lib.tensor.Tensor:
        seq_len = layer[0].shape()[2]
        assert seq_len <= self.max_seq_len_cached, "seq_len exceeds max_seq_len_cached in RotaryEmbedding!"
        # TODO: Make rotary embedding in place
        output = []
        for i in range(len(layer)):
            output.append(
                tt_lib.tensor.rotary_embedding(
                    layer[i],
                    self.tt_cos_cached[i],
                    self.tt_sin_cached[i],
                    token_idx,
                    output_mem_config=self.model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"],
                )
            )
        return output


class TtFalconAttention:
    """Mulit-Query Attention: https://arxiv.org/pdf/1911.02150.pdf"""

    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings: int = 2048,
        model_config=None,
        tt_cache_path=None,
        global_cos_sin_cache=None,
        emulate_per_device_fracture=False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings
        self.devices = devices
        self.state_dict = state_dict
        self.model_config = model_config
        self.emulate_per_device_fracture = emulate_per_device_fracture
        self.temperature = 0.92

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        num_devices = len(devices)
        # check if all devices are the same
        self.same_device = all(self.devices[i] == self.devices[i + 1] for i in range(num_devices - 1))

        layer_name = f"{base_url}.{layer_num}.self_attention"
        query_key_value_str = f"{layer_name}.query_key_value.weight"
        selfout_str = f"{layer_name}.dense.weight"

        self.query_key_value_weights = []
        self.dense_weights = []

        for i in range(num_devices):
            query_key_value_path = (
                tt_cache_path
                / f"{query_key_value_str}_{i}_{num_devices}_{self.model_config['FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (query_key_value_path).exists():
                self.query_key_value_weights.append(
                    tt_lib.tensor.load_tensor(str(query_key_value_path)).to(
                        devices[i], self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                query_key_value_weights_host = torch.transpose(
                    self.state_dict[query_key_value_str],
                    -2,
                    -1,
                )
                query_key_value_weights_host = torch2tt_tensor(
                    torch.chunk(query_key_value_weights_host, num_devices, -1)[i],
                    None,
                    tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
                )
                self.query_key_value_weights.append(
                    query_key_value_weights_host.to(devices[i], self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(query_key_value_path),
                    query_key_value_weights_host,
                )

            selfout_path = (
                tt_cache_path
                / f"{selfout_str}_{i}_{num_devices}_{self.model_config['SELFOUT_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (selfout_path).exists():
                self.dense_weights.append(
                    tt_lib.tensor.load_tensor(str(selfout_path)).to(
                        devices[i], self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                dense_weights_host = torch2tt_tensor(
                    torch.transpose(torch.chunk(self.state_dict[selfout_str], num_devices)[i], -2, -1),
                    None,
                    tt_memory_config=self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["SELFOUT_MM_WEIGHTS_DTYPE"],
                )
                self.dense_weights.append(
                    dense_weights_host.to(devices[i], self.model_config["SELFOUT_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(selfout_path),
                    dense_weights_host,
                )
        self.rotary_embedding = TtFalconRotaryEmbedding(
            self.devices,
            self.head_dim,
            base_url,
            layer_num=layer_num,
            max_position_embeddings=self.max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            global_cos_sin_cache=global_cos_sin_cache,
        )

        # Fused to SM
        # self.scalar = pad_by_zero(torch.Tensor([1 / math.sqrt(self.head_dim)]), self.device)[0]
        self.scalar = 1 / math.sqrt(self.head_dim)

    def __call__(
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

        assert not output_attentions

        if llm_mode == "prefill":
            # assert False, "Not supported"
            batch = hidden_states[0].shape()[0]
            q_len = hidden_states[0].shape()[2]
            rows_dim = q_len
            print(f"Running prefill with seqlen: {q_len}")
            assert layer_past is not None
            print(f"layer past: {layer_past}")
        elif llm_mode == "decode":
            batch = hidden_states[0].shape()[2]
            q_len = hidden_states[0].shape()[0]
            rows_dim = batch
            print(f"Running decode with users: {batch}")
            padded_layer_past_len = nearest_32(layer_past_len + 1)
            # We always store max_position_embeddings for kv_cache,
            # so we need separate variable to store the actual len of the kv_cache
            assert layer_past is not None
            assert layer_past_len > 0 and layer_past_len <= self.max_position_embeddings
        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        # Reshard
        print(f"Reshard")
        ln_attn_out_memcfg = memcfg_for_seqlen(self.model_config["LN_ATTN_OUTPUT_MEMCFG"], rows_dim)
        print(f"ln_attn_out_memcfg shard shape: {ln_attn_out_memcfg.shard_spec.shape}")
        fused_qkv_mm_input_memcfg = memcfg_for_seqlen(self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"], rows_dim)
        print(f"fused_qkv_mm_input_memcfg shard shape: {fused_qkv_mm_input_memcfg.shard_spec.shape}")
        if ln_attn_out_memcfg != fused_qkv_mm_input_memcfg:
            for i in range(len(hidden_states)):
                print(f"i: {i}")
                print(f"hidden_states[i].shape: {hidden_states[i].shape()}")
                hidden_states[i] = tt_lib.tensor.sharded_to_interleaved(
                    hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            for i in range(len(hidden_states)):
                hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                    hidden_states[i], sharded_mem_config=fused_qkv_mm_input_memcfg
                )

        #################
        ### FUSED QKV ###
        #################
        print(f"Fused QKV")
        fused_query_key_value = []
        for i in range(len(hidden_states)):
            fused_query_key_value.append(
                tt_lib.operations.primary.matmul_1d(
                    hidden_states[i],
                    self.query_key_value_weights[i],
                    program_config=self.model_config["QKV_MM_PROGCFG"],
                    output_mem_config=fused_qkv_mm_input_memcfg,
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                )
            )
        ###########
        ### TMs ###
        ###########
        print(f"TMs")
        fused_qkv_mm_output_memcfg = memcfg_for_seqlen(self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"], rows_dim)
        create_qkv_heads_input_memcfg = memcfg_for_seqlen(self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"], rows_dim)
        if fused_qkv_mm_output_memcfg != create_qkv_heads_input_memcfg:
            for i in range(len(fused_query_key_value)):
                fused_query_key_value[i] = tt_lib.tensor.sharded_to_interleaved(
                    fused_query_key_value[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            for i in range(len(fused_query_key_value)):
                fused_query_key_value[i] = tt_lib.tensor.interleaved_to_sharded(
                    fused_query_key_value[i], sharded_mem_config=create_qkv_heads_input_memcfg
                )
        query_layer = []
        key_layer = []
        value_layer = []
        create_qkv_heads_output_memcfg = memcfg_for_seqlen(
            self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"], rows_dim
        )
        for i in range(len(fused_query_key_value)):
            q_layer, k_layer, v_layer = tt_lib.tensor.nlp_create_qkv_heads(
                fused_query_key_value[i],
                num_heads=self.num_heads // len(self.devices),
                num_kv_heads=self.num_kv_heads // len(self.devices),
                transpose_k_heads=False,
                output_mem_config=create_qkv_heads_output_memcfg,
            )
            query_layer.append(q_layer)
            key_layer.append(k_layer)
            value_layer.append(v_layer)
            fused_query_key_value[i].deallocate(True)

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
        print(f"K cache update")
        if llm_mode == "prefill":
            # tt_lib.tensor.fill_cache(layer_past[0], key_layer, user_id)

            key_layer_interleaved = []
            for i in range(len(key_layer)):  # TODO: remove and use sharded inputs for matmul
                key_layer_interleaved.append(
                    tt_lib.tensor.sharded_to_interleaved(
                        key_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                    )
                )
            key_cache = tt_lib.tensor.concat(key_layer_interleaved, 1)
            for i in range(len(key_layer_interleaved)):  # TODO: remove and use sharded inputs for matmul
                key_layer_interleaved[i].deallocate(True)

        elif llm_mode == "decode":
            kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
            if kv_cache_memcfg.is_sharded():
                kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
                kv_cache_shard_shape[0] = layer_past[0][0].shape()[1] * padded_layer_past_len
                kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape
            # Update kv_cache in place
            for i in range(len(key_layer)):
                tt_lib.tensor.update_cache(layer_past[0][i], key_layer[i], layer_past_len)
                key_layer[i].deallocate(True)
            # key and value layers will have kv_seq_len padded to nearest 32
            for i in range(len(layer_past[0])):
                key_layer[i] = tt_lib.tensor.unpad(
                    layer_past[0][i],
                    [0, 0, 0, 0],
                    [
                        batch - 1,
                        self.num_kv_heads // len(self.devices) - 1,
                        padded_layer_past_len - 1,
                        self.head_dim - 1,
                    ],
                    output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                )
            for i in range(len(key_layer)):
                key_layer[i] = tt_lib.tensor.interleaved_to_sharded(key_layer[i], sharded_mem_config=kv_cache_memcfg)

        ######################
        ### PRE-SOFTMAX MM ###
        ######################
        print(f"Pre-softmax")
        # TODO: Sharded transpose could be in place???
        k_transposed_output_memcfg = memcfg_for_seqlen(self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"], rows_dim)
        key_layer_transposed = []
        for i in range(len(key_layer)):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=k_transposed_output_memcfg,
                )
            )
            key_layer[i].deallocate(True)

        # Q * KˆT MM
        attn_weights = []
        if llm_mode == "prefill":
            for i in range(len(query_layer)):  # TODO: remove and use sharded inputs for matmul
                query_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                    query_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            for i in range(len(key_layer_transposed)):  # TODO: remove and use sharded inputs for matmul
                key_layer_transposed[i] = tt_lib.tensor.sharded_to_interleaved(
                    key_layer_transposed[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )

            pre_softmax_mm_output_memcfg = self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"]
            pre_softmax_mm_output_shard_shape = pre_softmax_mm_output_memcfg.shard_spec.shape
            pre_softmax_mm_output_shard_shape[-2] = pre_softmax_mm_output_shard_shape[-2] * q_len
            pre_softmax_mm_output_memcfg.shard_spec.shape = pre_softmax_mm_output_shard_shape

            print(f"Before Q*KˆT")
            for i in range(len(query_layer)):
                attn_weights.append(
                    # tt_lib.tensor.matmul(
                    tt_lib.operations.primary.matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        # compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                        # output_mem_config=pre_softmax_mm_output_memcfg,
                        output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
                query_layer[i].deallocate(True)
                key_layer_transposed[i].deallocate(True)

            print(f"After Q*KˆT")
            print(f"len(attn_weights): {len(attn_weights)}")

            print(f"pre_softmax_mm_output_memcfg shard shape: {pre_softmax_mm_output_memcfg.shard_spec.shape}")
            print(f"attn_weights[0] shape: {attn_weights[0].shape()}")

            for i in range(len(attn_weights)):  # TODO: remove this and use as output memcfg above
                attn_weights[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_weights[i], sharded_mem_config=pre_softmax_mm_output_memcfg
                )

            # attn_weights = tt_lib.tensor.matmul(
            #     query_layer,
            #     key_layer_transposed,
            #     output_mem_config=pre_softmax_mm_output_memcfg,
            # )

        elif llm_mode == "decode":
            for i in range(len(query_layer)):
                attn_weights.append(
                    tt_lib.operations.primary.transformers.group_attn_matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
                query_layer[i].deallocate(True)
                key_layer_transposed[i].deallocate(True)
        ###############
        ### SOFTMAX ###
        ###############
        print(f"Softmax")
        # Is this needed
        # attn_weights = tt_lib.tensor.mul_unary(attn_weights, 1 / self.temperature)
        softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
        if llm_mode == "prefill":
            softmax_progcfg.block_w = q_len // 32
            # softmax_progcfg.block_h = q_len // 32
        elif llm_mode == "decode":
            softmax_progcfg.block_w = padded_layer_past_len // 32
        for i in range(len(attn_weights)):
            print(f"softmax_progcfg block_w: {softmax_progcfg.block_w}")
            # print(f"softmax_progcfg block_h: {softmax_progcfg.block_h}")
            print(f"attn_weights shape: {attn_weights[i].shape()}")
            print(f"attn_weights len: {len(attn_weights)}")
            print(f"attention_mask shape: {attention_mask[i].shape()}")
            print(f"attention_mask len: {len(attention_mask)}")
            print(f"self.scalar: {self.scalar}")
            attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                attn_weights[i],
                self.scalar,
                attention_mask[i],
                # program_config=self.model_config["SOFTMAX_PROGCFG"], # TODO: why don't we use the updated softmax_progcfg here?!
                program_config=softmax_progcfg,
                is_causal_mask=True,
            )

        ######################
        ### V CACHE UPDATE ###
        ######################
        print(f"V cache udpate")
        if llm_mode == "prefill":
            # TODO Implement fill_cache
            # tt_lib.tensor.fill_cache(layer_past[1], value_layer, user_id)

            value_layer_interleaved = []
            for i in range(len(value_layer)):  # TODO: remove and use sharded inputs for matmul
                value_layer_interleaved.append(
                    tt_lib.tensor.sharded_to_interleaved(
                        value_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                    )
                )
            value_cache = tt_lib.tensor.concat(value_layer_interleaved, 1)
            for i in range(len(value_layer_interleaved)):  # TODO: remove and use sharded inputs for matmul
                value_layer_interleaved[i].deallocate(True)

        elif llm_mode == "decode":
            # Update kv_cache in place
            for i in range(len(value_layer)):
                tt_lib.tensor.update_cache(layer_past[1][i], value_layer[i], layer_past_len)
                value_layer[i].deallocate(True)
            for i in range(len(layer_past[1])):
                value_layer[i] = tt_lib.tensor.unpad(
                    layer_past[1][i],
                    [0, 0, 0, 0],
                    [
                        batch - 1,
                        self.num_kv_heads // len(self.devices) - 1,
                        padded_layer_past_len - 1,
                        self.head_dim - 1,
                    ],
                    output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                )
            for i in range(len(value_layer)):
                value_layer[i] = tt_lib.tensor.interleaved_to_sharded(
                    value_layer[i], sharded_mem_config=kv_cache_memcfg
                )

        layer_present = layer_past if use_cache else None

        ########################
        ### POST-SOFTMAX MM ###
        ########################
        print(f"Post-softmax")
        attn_output = []
        if llm_mode == "prefill":
            post_softmax_mm_output_memcfg = memcfg_for_seqlen(
                self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"], rows_dim
            )
            # attn_output = tt_lib.tensor.matmul(
            #     attn_weights,
            #     value_layer,
            #     output_mem_config=post_softmax_mm_output_memcfg,
            # )

            for i in range(len(attn_weights)):  # TODO: remove and use sharded inputs for matmul
                attn_weights[i] = tt_lib.tensor.sharded_to_interleaved(
                    attn_weights[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            for i in range(len(value_layer)):  # TODO: remove and use sharded inputs for matmul
                value_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                    value_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )

            for i in range(len(attn_weights)):
                attn_output.append(
                    tt_lib.operations.primary.matmul(
                        attn_weights[i],
                        value_layer[i],
                        # output_mem_config=post_softmax_mm_output_memcfg,
                        output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                        output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
            attn_weights[i].deallocate(True)
            value_layer[i].deallocate(True)

            print(f"post_softmax_mm_output_memcfg shard shape: {post_softmax_mm_output_memcfg.shard_spec.shape}")
            print(f"attn_output[0] shape: {attn_output[0].shape()}")

            for i in range(len(attn_output)):  # TODO: remove this and use as output memcfg above
                attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_output[i], sharded_mem_config=post_softmax_mm_output_memcfg
                )

        elif llm_mode == "decode":
            post_softmax_mm_output_memcfg = memcfg_for_seqlen(
                self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"], rows_dim
            )
            for i in range(len(attn_weights)):
                attn_output.append(
                    tt_lib.operations.primary.transformers.group_attn_matmul(
                        attn_weights[i],
                        value_layer[i],
                        compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                        output_mem_config=post_softmax_mm_output_memcfg,
                        output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                    )
                )
            attn_weights[i].deallocate(True)
            value_layer[i].deallocate(True)
        #########################
        ### ATTENTION SELFOUT ###
        #########################
        print(f"Attention selfout")
        concat_heads_output_memcfg = memcfg_for_seqlen(self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"], rows_dim)
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=concat_heads_output_memcfg,
            )
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                attn_output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )

        if self.same_device:
            num_fractures = len(attn_output)
            attn_output_concat_list = []
            attention_all_gather_output_memcfg = memcfg_for_seqlen(
                self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"], rows_dim
            )
            for i in range(num_fractures):
                attn_output_concat = tt_lib.tensor.concat(attn_output, 3)
                attn_output_concat_list.append(
                    tt_lib.tensor.interleaved_to_sharded(
                        attn_output_concat, sharded_mem_config=attention_all_gather_output_memcfg
                    )
                )
            for i in range(num_fractures):
                attn_output[i].deallocate(True)
            attn_output = attn_output_concat_list
        else:
            attn_output = tt_lib.tensor.all_gather(
                attn_output,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )
            attention_all_gather_output_memcfg = memcfg_for_seqlen(
                self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"], rows_dim
            )
            for i in range(len(attn_output)):
                attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_output[i], sharded_mem_config=attention_all_gather_output_memcfg
                )

        selfout_mm_output_memcfg = memcfg_for_seqlen(self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"], rows_dim)
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.operations.primary.matmul_1d(
                attn_output[i],
                self.dense_weights[i],
                program_config=self.model_config["SELFOUT_MM_PROGCFG"],
                output_mem_config=selfout_mm_output_memcfg,
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            )

        # TODO: remove when fill_kv in here
        layer_present = ([key_cache], [value_cache])
        return attn_output, layer_present
