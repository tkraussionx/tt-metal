import torch.nn as nn
import tt_lib


class TtLinear(nn.Module):
    def __init__(
        self, weight: tt_lib.tensor.Tensor, bias: tt_lib.tensor.Tensor = None
    ) -> tt_lib.tensor.Tensor:
        super(TtLinear, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        w = tt_lib.tensor.transpose(self.weight)
        x = tt_lib.tensor.matmul(x, w)

        if self.bias is not None:
            x = tt_lib.tensor.bcast(
                x, self.bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
            )
        return x
