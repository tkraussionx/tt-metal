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

class TtCodeGenMLP(torch.nn.Module):
    def __init__(self, base_address, config: CodeGenConfig(), state_dict, device):
        super().__init__()
        # Get the weights


        self.weight_fc_in = state_dict[f"{base_address}.fc_in.weight"]
        self.weight_fc_out = state_dict[f"{base_address}.fc_out.weight"]

        self.intermediate_size = self.weight_fc_in.shape[-2]
        self.embed_dim = self.weight_fc_in.shape[-1]

        self.dropout = nn.Dropout(config.resid_pdrop)

        self.config = config
        self.device = device

        # Push weights to Tt device
        self.tt_weight_fc_in = torch2tt_tensor(
            self.weight_fc_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_weight_fc_out = torch2tt_tensor(
            self.weight_fc_out, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        # Load biases
        self.tt_bias_fc_in = torch2tt_tensor(
            state_dict[f"{base_address}.fc_in.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_bias_fc_out = torch2tt_tensor(
            state_dict[f"{base_address}.fc_out.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.fc_in = Linear(self.embed_dim, self.intermediate_size, self.tt_weight_fc_in, self.tt_bias_fc_in)

        self.fc_out = Linear(self.intermediate_size, self.embed_dim, self.tt_weight_fc_out, self.tt_bias_fc_out)


    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = codegen_gelu.tt_new_gelu(hidden_states, self.device)
        hidden_states = self.fc_out(hidden_states)
        pt_hidden_states = tt2torch_tensor(hidden_states)
        pt_hidden_states = self.dropout(pt_hidden_states)
        tt_hidden_states = torch_to_tt_tensor_rm(pt_hidden_states, self.device, put_on_device=False)

        return hidden_states
