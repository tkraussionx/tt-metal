import torch
from torch import nn



def assign_norm_weight(layernorm, state_dict, key_w):
    pass


def assign_linear_weights(linear, state_dict, key_w: str):
    linear.weight = nn.Parameter(state_dict[f"{key_w}.weight"])

    if linear.bias is not None:
        linear.bias = nn.Parameter(state_dict[f"{key_w}.bias"])



def assign_conv_weight(conv, state_dict, key_w:str):
    conv.weight = nn.Parameter(state_dict[f"{key_w}.weight"])
    if conv.bias is not None:
        conv.bias = nn.Parameter(state_dict[f"{key_w}.bias"])
