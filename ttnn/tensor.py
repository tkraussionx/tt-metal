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

    @property
    def layout(self: "Tensor") -> DataType:
        return self._tensor.layout()

    def __getitem__(self: "Tensor", slices) -> "Tensor":
        device = self._tensor.device
        tensor = copy_from_device(self)
        tensor = to_torch(tensor)
        tensor = tensor[slices]
        tensor = from_torch(tensor, dtype=self.dtype)
        tensor = copy_to_device(tensor, device)
        return tensor

    def __repr__(self: "Tensor") -> str:
        return str(self._tensor)


def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[DataType]=None,
) -> Tensor:
    return Tensor(ttl.tensor.Tensor(tensor, dtype))


def to_torch(tensor: Tensor) -> "torch.Tensor":
    ttl_tensor = tensor._tensor
    if ttl_tensor.storage_type() == ttl.tensor.StorageType.DEVICE:
        raise ValueError("Tensor cannot be on device when converting to torch!")
    if ttl_tensor.layout() != ttl.tensor.Layout.ROW_MAJOR:
        ttl_tensor = ttl_tensor.to(ttl.tensor.Layout.ROW_MAJOR)
    return ttl_tensor.to_torch()


def copy_to_device(tensor, device):
    return Tensor(tensor._tensor.to(device))


def copy_from_device(tensor):
    return Tensor(tensor._tensor.cpu())


def free(self: "Tensor") -> str:
    self._tensor.deallocate(force=True)


__all__ = [
    "DataType",
    "uint32",
    "float32",
    "bfloat16",
    "bfloat8_b",
    "Tensor",
    "from_tensor",
    "to_tensor",
    "copy_to_device",
    "copy_from_device",
    "free",
]
