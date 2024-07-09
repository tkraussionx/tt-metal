# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
import math


class ShiftedWindowAttentionV2(nn.Module):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            # self.proj = nn.Linear(dim, dim, bias=proj_bias)
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = self._get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    # def get_relative_position_bias(self) -> torch.Tensor:
    #     return self._get_relative_position_bias(
    #         self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
    #     )

    def forward(self, x: Tensor):
        relative_position_bias = self.get_relative_position_bias()
        B, H, W, C = input.shape
        # pad feature maps to multiples of window size
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        shift_size = shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            shift_size[1] = 0

        # cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = x.view(
            B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B * num_windows, self.window_size[0] * self.window_size[1], C
        )  # B*nW, Ws*Ws, C

        # multi-head attention
        if logit_scale is not None and qkv_bias is not None:
            qkv_bias = qkv_bias.clone()
            length = qkv_bias.numel() // 3
            qkv_bias[length : 2 * length].zero_()
        qkv = F.linear(x, self.qkv_weight, qkv_bias)
        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if logit_scale is not None:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            q = q * (C // self.num_heads) ** -0.5
            attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -shift_size[0]), (-shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -shift_size[1]), (-shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(
                pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1]
            )
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attention_dropout, training=self.training)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = F.linear(x, self.proj_weight, self.proj_bias)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # reverse windows
        x = x.view(
            B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        return x
