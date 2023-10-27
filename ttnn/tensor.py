from typing import Optional

import tt_lib as ttl


DataType = ttl.tensor.DataType
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B


BufferType = ttl.tensor.BufferType
TensorMemoryLayout = ttl.tensor.TensorMemoryLayout
MemoryConfig = ttl.tensor.MemoryConfig
DRAM_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
L1_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)


Layout = ttl.tensor.Layout
ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE


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
        if self._tensor.storage_type() == ttl.tensor.StorageType.DEVICE:
            device = self._tensor.device()

            tensor = self
            tensor = to_layout(tensor, ROW_MAJOR_LAYOUT)
            tensor = from_device(tensor)
            tensor = to_torch(tensor)
            tensor = tensor[slices]
            tensor = from_torch(tensor, dtype=self.dtype)
            tensor = to_device(tensor, device)
        else:
            tensor = to_torch(self)
            tensor = tensor[slices]
            tensor = from_torch(tensor, dtype=self.dtype)
        return tensor

    def __repr__(self: "Tensor") -> str:
        return str(self._tensor)


def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[DataType] = None,
) -> Tensor:
    return Tensor(ttl.tensor.Tensor(tensor, dtype))


def to_torch(tensor: Tensor) -> "torch.Tensor":

    ttl_tensor = tensor._tensor

    if ttl_tensor.layout() != ROW_MAJOR_LAYOUT:
        raise RuntimeError("ttnn.Tensor has to be in ROW_MAJOR Layout to be convered to torch.Tensor")

    if ttl_tensor.storage_type() == ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch!")

    if ttl_tensor.layout() != ROW_MAJOR_LAYOUT:
        ttl_tensor = ttl_tensor.to(ROW_MAJOR_LAYOUT)
    return ttl_tensor.to_torch()


def to_device(tensor, device, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG):
    ttl_tensor = tensor._tensor
    return Tensor(
        ttl_tensor.to(device, memory_config)
    )


def from_device(tensor):
    ttl_tensor = tensor._tensor
    return Tensor(ttl_tensor.cpu())


def to_layout(tensor, layout: Layout):
    ttl_tensor = tensor._tensor
    if ttl_tensor.layout() == layout:
        return tensor
    elif layout == ROW_MAJOR_LAYOUT:
        ttl_tensor = ttl.tensor.untilize(ttl_tensor)
    elif layout == TILE_LAYOUT:
        ttl_tensor = ttl.tensor.tilize(ttl_tensor, output_mem_config=ttl_tensor.memory_config())
    return Tensor(ttl_tensor)


def free(tensor: Tensor) -> str:
    tensor._tensor.deallocate(force=True)


__all__ = [
    "DataType",
    "uint32",
    "float32",
    "bfloat16",
    "bfloat8_b",

    "DRAM_MEMORY_CONFIG",
    "L1_MEMORY_CONFIG",

    "ROW_MAJOR_LAYOUT",
    "TILE_LAYOUT",

    "Tensor",
    "from_tensor",
    "to_tensor",
    "to_device",
    "from_device",
    "to_layout",
    "free",
]
