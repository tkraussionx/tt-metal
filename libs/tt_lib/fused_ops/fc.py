
import tt_lib as ttl

from typing import Union, List


def run_fc_on_device_wrapper(weight: List[Union[int, float]], params, device, bias=None):
    ## NOTE: Assuming weight is padded and in RM, bias is 1d padded (also RM)
    weight_shape, weight_dtype, bias_shape, bias_dtype = params
    fc_weight = ttl.tensor.Tensor(weight, weight_shape, weight_dtype, ttl.tensor.Layout.ROW_MAJOR)
    fc_weight = fc_weight.to(ttl.tensor.Layout.TILE).to(device)
    fc_weight = ttl.tensor.transpose(fc_weight)
    fc_bias = None
    if bias:
        fc_bias = ttl.tensor.Tensor(bias, bias_shape, bias_dtype, ttl.tensor.Layout.ROW_MAJOR)
        fc_bias = fc_bias.to(device)

    def fc(x):
        out = ttl.tensor.fully_connected(x, fc_weight, fc_bias)
        return out

    return fc
