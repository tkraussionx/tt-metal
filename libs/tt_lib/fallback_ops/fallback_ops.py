import torch
from .conversion_wrapper import convert_tt_tensors_wrapper


@convert_tt_tensors_wrapper
def full(size, fill_value):
    return torch.full(size, fill_value)


@convert_tt_tensors_wrapper
def reshape(input, N, C, H, W):
    return torch.reshape(input, (N, C, H, W))


@convert_tt_tensors_wrapper
def chunk(input, chunks, dim=0):
    return torch.chunk(input, chunks, dim)


@convert_tt_tensors_wrapper
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return torch.nn.functional.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )


@convert_tt_tensors_wrapper
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    return torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)


@convert_tt_tensors_wrapper
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


@convert_tt_tensors_wrapper
def pad(input, pad, mode="constant", value=None):
    return torch.nn.functional.pad(input, pad, mode, value)


@convert_tt_tensors_wrapper
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    return torch.repeat_interleave(input, repeats, dim, output_size=output_size)


@convert_tt_tensors_wrapper
def softmax(input, dim=None):
    return torch.nn.functional.softmax(input, dim)


class Conv2d(torch.nn.Conv2d):
    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)


class GroupNorm(torch.nn.GroupNorm):
    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)


class LayerNorm(torch.nn.LayerNorm):
    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)


class Softmax(torch.nn.Softmax):
    @convert_tt_tensors_wrapper
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)
