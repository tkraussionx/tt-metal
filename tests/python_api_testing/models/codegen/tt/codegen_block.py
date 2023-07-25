import torch
from torch.nn import functional as F

import tt_lib
from python_api_testing.models.helper_funcs import Linear
import python_api_testing.models.codegen.tt.codegen_gelu as codegen_gelu
import python_api_testing.models.codegen.tt.codegen_attention as codegen_attention
import python_api_testing.models.codegen.tt.codegen_mlp as codegen_mlp

from torch import nn
from functools import partial
from tt_lib.fallback_ops import fallback_ops

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import CodeGenConfig, CodeGenModel

class TtCodeGenBlock(torch.nn.Module):
    def __init__(self, block, config: CodeGenConfig(), state_dict, device):
        super().__init__()
        # Get the weights
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        base_address = f"h.{block}"

        self.device = device

        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_1.weight"], self.device, put_on_device=False
        )
        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_1.bias"], self.device, put_on_device=False
        )

        self.ln_1 = partial(
            tt_lib.tensor.layernorm,
            gamma=self.gamma,
            beta=self.beta,
            eps=config.layer_norm_epsilon,
        )


        base_address_mlp = f"h.{block}.mlp"

        base_address_attn = f"h.{block}.attn"

        self.attn = codegen_attention.TtCodeGenAttention(base_address_attn, config, state_dict, device)

        self.mlp = codegen_mlp.TtCodeGenMLP(base_address_mlp, config, state_dict, device)


    def forward(
        self,
        device,
        hidden_states,
        layer_past = None,
        attention_mask = None,
        head_mask = None,
        use_cache = False,
        output_attentions = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            device,
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)

        attn_output_shape = attn_output.shape()
        print('ATTN out shape')

        print(attn_output_shape)


        pt_attn_output = tt2torch_tensor(attn_output)
        pt_attn_output = pt_attn_output.squeeze(0)
        pt_attn_output = pt_attn_output.squeeze(0)
        pt_attn_output = pt_attn_output.squeeze(0)

        outputs = pt_attn_output[1:]



        #slice_list_outputs = [None, None, slice(1, attn_output_shape[3])]

        #outputs = fallback_ops.tensor_slice(attn_output, slice_list_outputs)

        attn_output = torch_to_tt_tensor_rm(outputs, device, put_on_device=False)

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = tt_lib.tensor.add(attn_output, feed_forward_hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, residual)


        outputs_shape = outputs.shape()

        slice_list_outputs_2 = [slice(1, outputs_shape[0])]

        outputs_sliced = fallback_ops.tensor_slice(outputs, slice_list_outputs_2)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs_sliced

        return outputs  # hidden_states, present, (attentions)
