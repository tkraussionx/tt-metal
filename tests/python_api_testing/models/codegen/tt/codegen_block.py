import torch
from torch.nn import functional as F

import tt_lib
from python_api_testing.models.helper_funcs import Linear
import python_api_testing.models.codegen.tt.codegen_gelu as codegen_gelu
from torch import nn

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import CodeGenConfig, CodeGenModel

class TtCodeGenBlock(torch.nn.Module):
    def __init__(self, base_address, config: CodeGenConfig(), state_dict, device):
        super().__init__()
        # Get the weights
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd

        self.beta = bloom_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.bias"], device
        )

        self.gamma = bloom_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.weight"], device
        )

        self.ln_1 = fallback_ops.LayerNorm(
            self.gamma,
            self.beta,
            eps=config.layer_norm_epsilon,
            normalized_shape=config.hidden_size,
        )

        self.attn = TtCodeGenAttention(config)
        self.mlp = TtCodeGenMLP(inner_dim, config)

    def forward(
        self,
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

        slice_list_outputs = [slice(1, attn_output_shape[0])]

        outputs = fallback_ops.tensor_slice(attn_output, slice_list_outputs)

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
