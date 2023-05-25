import torch
from torch.nn import functional as F

import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_gelu as nanogpt_gelu



class MLP(torch.nn.Module):

    def __init__(self, config, index):
        super().__init__()
        self.si = str(index)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        ddict["mlp_cfc"+self.si] = x.clone()
        x = new_gelu(x)
        ddict["mlp_gelu"+self.si] = x.clone()
        x = self.c_proj(x)
        ddict["mlp_cproj"+self.si] = x.clone()
        x = self.dropout(x)
        return x

class TtMLP(torch.nn.Module):
    def __init__(self, config, state_dict, index, device):
        super().__init__()

        self.si = str(index)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)



        # Get the weights
        self.tt_weight_c_fc = state_dict[f"{index}.c_fc"]
        self.tt_weight_c_proj = state_dict[f"{index}.c_proj"]

        # Transpose the weights
        self.tt_weight_mlp_h4h = torch.transpose(self.tt_weight_mlp_h4h, -1, -2)
        self.tt_weight_mlp_4hh = torch.transpose(self.tt_weight_mlp_4hh, -1, -2)

        # Push weights to Tt device
        self.tt_weight_mlp_h4h = bloom_utils.torch2tt_tensor(self.tt_weight_mlp_h4h, device)
        self.tt_weight_mlp_4hh = bloom_utils.torch2tt_tensor(self.tt_weight_mlp_4hh, device)

        # Load biases
        self.tt_bias_mlp_h4h = bloom_utils.torch2tt_tensor(state_dict[f"{index}.dense_h_to_4h.bias"], device)
        self.tt_bias_mlp_4hh = bloom_utils.torch2tt_tensor(state_dict[f"{index}.dense_4h_to_h.bias"], device)

        # self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward
        self.gelu_impl = bloom_gelu_forward.bloom_gelu_forward






        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.training = False

        # Get the weights
        self.tt_weight_mlp_h4h = state_dict[f"{index}.dense_h_to_4h.weight"]
        self.tt_weight_mlp_4hh = state_dict[f"{index}.dense_4h_to_h.weight"]

        # Transpose the weights
        self.tt_weight_mlp_h4h = torch.transpose(self.tt_weight_mlp_h4h, -1, -2)
        self.tt_weight_mlp_4hh = torch.transpose(self.tt_weight_mlp_4hh, -1, -2)

        # Push weights to Tt device
        self.tt_weight_mlp_h4h = bloom_utils.torch2tt_tensor(self.tt_weight_mlp_h4h, device)
        self.tt_weight_mlp_4hh = bloom_utils.torch2tt_tensor(self.tt_weight_mlp_4hh, device)

        # Load biases
        self.tt_bias_mlp_h4h = bloom_utils.torch2tt_tensor(state_dict[f"{index}.dense_h_to_4h.bias"], device)
        self.tt_bias_mlp_4hh = bloom_utils.torch2tt_tensor(state_dict[f"{index}.dense_4h_to_h.bias"], device)

        # self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward
        self.gelu_impl = bloom_gelu_forward.bloom_gelu_forward

    def forward(self, hidden_states, residual, device):

        # h4h = self.dense_h_to_4h(hidden_states)
        h4h = bloom_utils.tt_matmul(hidden_states, self.tt_weight_mlp_h4h, device)
        h4h = tt_lib.tensor.bcast(h4h, self.tt_bias_mlp_h4h, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)
        h4h = bloom_utils.tt2torch_tensor(h4h)

        hidden_states = self.gelu_impl(h4h)

        hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)
        intermediate_output = bloom_utils.tt_matmul(hidden_states, self.tt_weight_mlp_4hh, device)
        intermediate_output = tt_lib.tensor.bcast(intermediate_output, self.tt_bias_mlp_4hh, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)

        # Dropout is used in training only
        # intermediate_output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        output = tt_lib.tensor.add(residual, intermediate_output)

        return output
