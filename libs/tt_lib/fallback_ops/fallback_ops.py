import torch
from .conversion_wrapper import convert_tt_tensors_wrapper


@convert_tt_tensors_wrapper
def full(size, fill_value):
    """
    Creates a ``ttlib.tensor.Tensor`` of size ``size`` filled with ``fill_value``.
    """
    return torch.full(size, fill_value)


@convert_tt_tensors_wrapper
def reshape(input, N, C, H, W):
    """
    Returns a new ``ttlib.tensor.Tensor`` with the same data and number of elements as ``input``, but with the specified shape.
    """
    return torch.reshape(input, (N, C, H, W))


@convert_tt_tensors_wrapper
def chunk(input, chunks, dim=0):
    """
    Attempts to split a ``ttlib.tensor.Tensor`` into the specified number of chunks. Each chunk is a new copy of part of the input tensor.
    """
    return torch.chunk(input, chunks, dim)


@convert_tt_tensors_wrapper
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    """
    return torch.nn.functional.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )


@convert_tt_tensors_wrapper
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    """
    Applies Group Normalization for last certain number of dimensions.
    """
    return torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)


@convert_tt_tensors_wrapper
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """
    Applies Layer Normalization for last certain number of dimensions.
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


@convert_tt_tensors_wrapper
def pad(input, pad, mode="constant", value=None):
    """
    Pads tensor.
    """
    return torch.nn.functional.pad(input, pad, mode, value)

@convert_tt_tensors_wrapper
def silu(input):
    return torch.nn.functional.silu(input)

@convert_tt_tensors_wrapper
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    """
    Repeat elements of a tensor.
    """
    return torch.repeat_interleave(input, repeats, dim, output_size=output_size)

@convert_tt_tensors_wrapper
def concat(tensors, dim=0):
    """
    Concatenates the given sequence of ``seq`` tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    """
    return torch.concat(tensors, dim)

@convert_tt_tensors_wrapper
def silu(input):
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    """
    return torch.nn.functional.silu(input)

@convert_tt_tensors_wrapper
def softmax(input, dim=None):
    r"""
    Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    """
    return torch.nn.functional.softmax(input, dim)


class Conv2d(torch.nn.Conv2d):
    r"""
    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    """

    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)


class GroupNorm(torch.nn.GroupNorm):
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.
    """

    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)


class LayerNorm(torch.nn.LayerNorm):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.
    """
    @convert_tt_tensors_wrapper
    def __init__(self, weights, biases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)

class SiLU(torch.nn.SiLU):
    @convert_tt_tensors_wrapper
    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)

class SiLU(torch.nn.SiLU):
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    """
    @convert_tt_tensors_wrapper
    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)

class Softmax(torch.nn.Softmax):
    r"""
    Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """
    @convert_tt_tensors_wrapper
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @convert_tt_tensors_wrapper
    def forward(self, input):
        return super().forward(input)
