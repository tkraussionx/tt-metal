# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import tt_lib
import math
from models.helper_funcs import Linear
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.experimental.nanogpt.nanogpt_helper_funcs import format_tensor, unpad_from_zero


class TtCausalSelfAttention(nn.Module):
    def __init__(self, config, base_address, device, tt_cache_path, dtype):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.config = config
        self.block_size = 1024

        self.device = device
        self.output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )

        # Get the weights
        self.tt_weight_c_attn = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_attn.weight" + str(dtype) + ".bin"
        )

        self.tt_weight_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".bin"
        )

        self.tt_weight_c_attn = tt_lib.tensor.transpose(self.tt_weight_c_attn, -2, -1)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj, -2, -1)

        # Load biases
        self.tt_bias_c_attn = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_attn.bias" + str(dtype) + ".bin"
        )

        self.tt_bias_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".bin"
        )

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        temp_bias = tt_lib.tensor.tril(tt_lib.tensor.ones([1, 1, self.block_size, self.block_size]))
        temp_bias = tt_to_torch_tensor(temp_bias)
        self.register_buffer(
            "bias",
            temp_bias,
        )

        self.c_attn = Linear(
            self.config.n_embd,
            3 * config.n_embd,
            self.tt_weight_c_attn,
            self.tt_bias_c_attn,
        )
        self.c_proj = Linear(
            self.config.n_embd,
            self.config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
        )

    def const_tensor(self, shape, value):
        return tt_lib.tensor.full(shape, value)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        (
            _,
            B,
            T,
            C,
        ) = x.shape()  # batch size, sequence length, embedding dimensionality (n_embd)

        desired_x1_shape = x.shape().copy()
        x1 = self.c_attn(x)

        desired_x1_shape[-1] = self.tt_weight_c_attn.shape()[-2]

        pt_x1 = unpad_from_zero(x1, desired_x1_shape)
        pt_x1 = pt_x1.squeeze(0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = pt_x1.split(self.n_embd, dim=2)

        k = torch_to_tt_tensor_rm(k, self.device)
        k = tt_lib.tensor.reshape(k, B, T, self.n_head, C // self.n_head)
        k = tt_lib.tensor.transpose(k, 1, 2)

        q = torch_to_tt_tensor_rm(q, self.device)
        q = tt_lib.tensor.reshape(q, B, T, self.n_head, C // self.n_head)
        q = tt_lib.tensor.transpose(q, 1, 2)

        v = torch_to_tt_tensor_rm(v, self.device)
        v = tt_lib.tensor.reshape(v, B, T, self.n_head, C // self.n_head)
        v = tt_lib.tensor.transpose(v, 1, 2)

        # manual implementation of attention
        k = format_tensor(k, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        q = format_tensor(q, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        v = format_tensor(v, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        key_layer_transposed = tt_lib.tensor.transpose(k, -2, -1)

        att = tt_lib.tensor.bmm(q, key_layer_transposed)

        desired_att_shape = key_layer_transposed.shape().copy()
        desired_att_shape[-1] = desired_att_shape[-2] = T

        const_att = self.const_tensor(att.shape(), 1.0 / math.sqrt(k.shape()[-1]))

        const_att = format_tensor(const_att, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        att = tt_lib.tensor.mul(att, const_att)

        att = unpad_from_zero(att, desired_att_shape)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        tt_att = torch_to_tt_tensor_rm(att, self.device, put_on_device=True)

        tt_att = format_tensor(tt_att, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        tt_att = tt_lib.tensor.softmax(tt_att)

        tt_y = tt_lib.tensor.bmm(tt_att, v)

        tt_y = format_tensor(tt_y, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        tt_y = tt_lib.tensor.transpose(tt_y, 1, -2)

        tt_y = tt_lib.tensor.reshape(tt_y, 1, B, T, C)

        # output projection
        tt_y = format_tensor(tt_y, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        x2 = self.c_proj(tt_y)
        return x2
