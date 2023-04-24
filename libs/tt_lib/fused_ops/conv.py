from typing import List, Union
from .. import tensor
from libs.tt_lib.utils import _nearest_32

def conv(weight: List[Union[int, float]], conv_params, device, bias=None):
    """
    Returns a function that performs a Convolution.
    bias is optional. If provided, it must be in tiled layout
    """
    assert(len(conv_params) == 8)
    K, C, R, S, U, V, P_H, P_W = [conv_params[i] for i in range(8)]

    weight_untiled = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    assert(weight_tiled_.shape() == [1, 1, C*R*S, K])
    weight = weight_tiled_.to(device)

    if bias is None:
        bias = None
    else:
        bias = tensor.Tensor(
            bias,
            [1, 1, 1, K],
            tensor.DataType.BFLOAT16,
            tensor.Layout.ROW_MAJOR,
            device
        )

    def conv_(activation):
        # check if params are valid
        [N,C,H,W] = activation.shape()
        assert (H - R + 2 * P_H) >= 1 and (W - S + 2 * P_W) >= 1
        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        conv_as_mm_output_shape = [1,1,_nearest_32(OH*OW),K]
        out = ttl.tensor.conv_as_large_bmm_single_core(activation, weight, [R,S,U,V,P_H,P_W])
        assert(out.shape() == conv_as_mm_output_shape)

        if bias is not None:
            assert False # unsupported
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return conv_
