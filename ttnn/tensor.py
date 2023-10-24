from typing import Optional

import tt_lib as ttl


DataType = ttl.tensor.DataType
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B


class Tensor:
    def __init__(self: "Tensor", ttl_tensor: ttl.tensor.Tensor):
        self._tensor: ttl.tensor.Tensor = ttl_tensor

    @property
    def shape(self: "Tensor") -> tuple:
        return self._tensor.shape()

    @property
    def dtype(self: "Tensor") -> DataType:
        return self._tensor.dtype()

    def __getitem__(self: "Tensor", slices) -> "Tensor":
        torch_tensor = to_torch(self)
        torch_tensor = torch_tensor[slices]
        return from_torch(torch_tensor, dtype=self.dtype)

    def __repr__(self: "Tensor") -> str:
        return str(self._tensor)


def from_torch(
    torch_tensor: "torch.Tensor",
    dtype: Optional[DataType]=None,
) -> Tensor:
    return Tensor(ttl.tensor.Tensor(torch_tensor, dtype))


def to_torch(tt_tensor: Tensor) -> "torch.Tensor":
    tt_output = tt_tensor._tensor.cpu() # Move to CPU if on device
    if tt_output.layout() != ttl.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(ttl.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()

__all__ = [
    "DataType",
    "Tensor",
    "from_tensor",
    "to_tensor",
]
