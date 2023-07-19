import torch
from torch.nn import functional as F

import tt_lib
from python_api_testing.models.helper_funcs import Linear
import python_api_testing.models.codegen.tt.codegen_gelu as codegen_gelu
import python_api_testing.models.codegen.tt.codegen_merge_heads as codegen_merge_heads
import python_api_testing.models.codegen.tt.codegen_split_heads as codegen_split_heads

import python_api_testing.models.codegen.tt.codegen_fixed_pos_emb as codegen_fixed_pos_emb
import python_api_testing.models.codegen.tt.codegen_rotary_pos_emb as codegen_rotary_pos_emb

from tt_lib.fallback_ops import fallback_ops

from torch import nn

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import CodeGenConfig, CodeGenModel

class TtCodeGenAttention(nn.Module):
    def __init__(self, base_address, config: CodeGenConfig(), state_dict, device):
        super().__init__()

        max_positions = config.max_position_embeddings

        self.causal_mask = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions)

        """
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        """

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        print(self.embed_dim)
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )


        #self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.weight_qkv_proj = state_dict[f"{base_address}.qkv_proj.weight"]
        self.weight_out_proj = state_dict[f"{base_address}.out_proj.weight"]


        self.embed_dim = self.weight_qkv_proj.shape[-1]

         # Push weights to Tt device
        self.tt_weight_qkv_proj = torch2tt_tensor(
            self.weight_qkv_proj, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.tt_weight_out_proj = torch2tt_tensor(
            self.weight_out_proj, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )



        # Load biases
        """
        self.tt_bias_qkv_proj = torch2tt_tensor(
            state_dict[f"{base_address}.qkv_proj.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        """
        """
        self.tt_bias_out_proj = torch2tt_tensor(
            state_dict[f"{base_address}.out_proj.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        """

        self.qkv_proj = Linear(self.embed_dim, self.embed_dim * 3, self.tt_weight_qkv_proj, None) #self.tt_bias_qkv_proj)


        self.out_proj = Linear(self.embed_dim, self.embed_dim, self.tt_weight_out_proj, None) #self.tt_bias_out_proj)


        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim


    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = torch.Size(query.shape())(-2), torch.Size(key.shape())(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        #query = query.to(torch.float32)
        #key = key.to(torch.float32)

        tt_key_transposed = tt_lib.tensor.transpose(key)

        tt_attn_weights = tt_lib.tensor.matmul(query, tt_key_transposed)

        tt_scale_attn_recip = tt_lib.tensor.recip(self.scale_attn)

        tt_attn_weights = tt_lib.tensor.matmul(tt_attn_weights, tt_scale_attn_recip)

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        #mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)

        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        t3 = ttl.tensor.where(t0, t1, t2)

        if attention_mask is not None:
            # Apply the attention mask
            tt_attn_weights = tt_lib.tensor.add(tt_attn_weights, tt_attention_mask)

        tt_attn_weights = fallback_ops.softmax(tt_attn_weights, dim=-1)

        #attn_weights = attn_weights.to(value.dtype)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            tt_attn_weights = tt_lib.tensor.matmul(attn_weights, head_mask)

        tt_attn_output = tt_lib.tensor.matmul(tt_attn_weights, value)

        return tt_attn_output, tt_attn_weights


    def forward(
        self,
        device,
        hidden_states,
        attention_mask = None,
        layer_past = None,
        head_mask  = None,
        use_cache = False,
        output_attentions = False,
    ):
        print()
        qkv = self.qkv_proj(hidden_states)
        mp_num = 4

        print(qkv.shape())
        pt_qkv = tt2torch_tensor(qkv)
        print(pt_qkv.shape)
        pt_qkv_split = pt_qkv.reshape(pt_qkv.shape[:-1] + (mp_num, -1))

        #qkv_split = torch2tt_tensor(pt_qkv_split, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        #qkv_split = tt_lib.tensor.reshape(qkv, qkv_new_shape[0], qkv_new_shape[1], qkv_new_shape[2], qkv_new_shape[3])

        local_dim = self.head_dim * self.num_attention_heads // mp_num

        res = torch.split(pt_qkv_split, local_dim, dim=-1)
        print(len(res))
        pt_query = res[0]
        pt_value = res[1]
        pt_key = res[2]


        query = torch2tt_tensor(pt_query, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        value = torch2tt_tensor(pt_value, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        key = torch2tt_tensor(pt_key, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)


        query = codegen_split_heads.tt_split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = codegen_split_heads.tt_split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = codegen_split_heads.tt_split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape()
            offset = offset[-2]
            seq_len += offset

        if self.rotary_dim is not None:


            query_shape = query.shape()
            key_shape = key.shape()


            slice_list_1 = [slice(None), slice(None), slice(None), slice(0, self.rotary_dim)]
            slice_list_2 = slice(None), slice(None), slice(None), slice(self.rotary_dim, query_shape[3])
            slice_list_3 = slice(None), slice(None), slice(None), slice(self.rotary_dim, key_shape[3])


            k_rot = fallback_ops.tensor_slice(key, slice_list_1)
            k_pass = fallback_ops.tensor_slice(key, slice_list_3)


            q_rot = fallback_ops.tensor_slice(query, slice_list_1)
            q_pass = fallback_ops.tensor_slice(query, slice_list_2)

            sincos = codegen_fixed_post_emb.tt_fixed_pos_embedding(k_rot, device, 1, seq_len=seq_len)
            k_rot = codegen_rotary_pos_emb.tt_rotary_pos_emb(k_rot, device, sincos, offset=offset)
            q_rot = codegen_rotary_pos_emb.tt_rotary_pos_emb(q_rot, device, sincos, offset=offset)

            key = fallback_ops.concat([k_rot, k_pass], dim=-1)
            query = fallback_ops.concat([q_rot, q_pass], dim=-1)

        else:
            sincos = codegen_fixed_pos_emb.tt_fixed_pos_emb(key, device, 1, seq_len=seq_len)
            key = codegen_rotary_pos_emb.tt_rotary_pos_emb(key, device, sincos, offset=offset)
            query = codegen_apply_rotary_pos_emb.tt_rotary_pos_emb(query, device, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = fallback_ops.concat((past_key, key), dim=-2)
            value = fallback_ops.concat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = codegen_merge_heads.tt_merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
