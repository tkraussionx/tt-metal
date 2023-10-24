from loguru import logger

import tt_lib as ttl



class Tensor:
    def __init__(self, ttl_tensor):
        self._tensor = ttl_tensor

    @property
    def shape(self):
        return self._tensor.shape()

    @property
    def dtype(self):
        return self._tensor.dtype()

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, slices):
        torch_tensor = to_torch(self)
        torch_tensor = torch_tensor[slices]
        return from_torch(torch_tensor, dtype=self.dtype)

    def __repr__(self):
        return str(self._tensor)


def shape(tensor):
    return ttl.tensor.shape(tensor)


def is_scalar(value):
    return isinstance(value, (int, float, complex))


# Conversion
def to_torch(tt_tensor):
    tt_output = tt_tensor._tensor.cpu()
    if tt_output.layout() != ttl.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(ttl.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()


def from_torch(
    torch_tensor,
    dtype=None,
):
    return Tensor(ttl.tensor.Tensor(torch_tensor, dtype))


def _shape_is_broadcastable(input_shape_a, input_shape_b):
    *batch_shape_a, height_a, width_a = input_shape_a
    *batch_shape_b, height_b, width_b = input_shape_b

    # if width_a != height_b:
    #     return False

    len_diff = len(batch_shape_a) - len(batch_shape_b)
    if len_diff > 0:
        batch_shape_b = [1] * len_diff + batch_shape_b
    else:
        batch_shape_a = [1] * -len_diff + batch_shape_a

    return all(x == y or (x == 1 and y != 1) or (x != 1 and y == 1) for x, y in zip(batch_shape_a, batch_shape_b))


# Math Operations
# Should the matmal autodetect if the tensor is on device?
#   * Should one type of operation be prefered over the other for optimizations?
def matmul(input_tensor_a, input_tensor_b):

    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a tt_lib.tensor.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_a = input_tensor_a.shape
    input_shape_b = input_tensor_b.shape

    *_, height_a, width_a = input_shape_a
    *rest_of_shape_b, height_b, width_b = input_shape_b

    # The idea is to make the shapes "possibly" broadcastable.
    len_diff = len(input_shape_a) - len(input_shape_b)
    if len_diff > 0:
        input_shape_b = [1] * len_diff + input_shape_b
        input_tensor_b = reshape(input_tensor_b, shape=input_shape_b)
    else:
        input_shape_a = [1] * -len_diff + input_shape_a
        input_tensor_a = reshape(input_tensor_a, shape=input_shape_a)

    # if height_a % 32 != 0 or width_a % 32 != 0:
    #     raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

    # if height_b % 32 != 0 or width_b % 32 != 0:
    #     raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

    if width_a != height_b and height_a != 1 and height_b != 1:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    if height_b == 1 and width_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a._tensor, input_tensor_b._tensor, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW))
    elif _shape_is_broadcastable(input_shape_a, input_shape_b):
        if all(x == 1 for x in rest_of_shape_b):
            if width_a != height_b:
                raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")
            return Tensor(ttl.tensor.matmul(input_tensor_a._tensor, input_tensor_b._tensor))
        else:
            return Tensor(ttl.tensor.bmm(input_tensor_a._tensor, input_tensor_b._tensor))
    else:
        raise RuntimeError("These tensors cannot be broadcasted")


def add(input_tensor_a, input_tensor_b, *, alpha=1):

    input_tensor_a = input_tensor_a._tensor if isinstance(input_tensor_a, Tensor) else input_tensor_a
    input_tensor_b = input_tensor_b._tensor if isinstance(input_tensor_b, Tensor) else input_tensor_b

    if not isinstance(input_tensor_a, ttl.tensor.Tensor):
        raise TypeError("Expected first argument to be a tt_lib.tensor.Tensor")

    input_shape_a = input_tensor_a.shape()

    if is_scalar(input_tensor_b):
        return Tensor(ttl.tensor.add_unary(input_tensor_a, input_tensor_b * alpha))
    elif not isinstance(input_tensor_b, ttl.tensor.Tensor):
        raise TypeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_b = input_tensor_b.shape()

    if alpha != 1:
        input_tensor_b = ttl.tensor.mul_unary(input_tensor_b, alpha)

    *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW))
    elif height_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H))
    elif width_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W))
    return Tensor(ttl.tensor.add(input_tensor_a, input_tensor_b))


def subtract(input_tensor_a, input_tensor_b):

    input_tensor_a = input_tensor_a._tensor
    input_tensor_b = input_tensor_b._tensor

    return Tensor(ttl.tensor.sub(input_tensor_a, input_tensor_b))


def mul(input_tensor_a, input_tensor_b):

    input_tensor_a = input_tensor_a._tensor if isinstance(input_tensor_a, Tensor) else input_tensor_a
    input_tensor_b = input_tensor_b._tensor if isinstance(input_tensor_b, Tensor) else input_tensor_b

    if not isinstance(input_tensor_a, ttl.tensor.Tensor):
        raise TypeError("Expected first argument to be a tt_lib.tensor.Tensor")

    input_shape_a = input_tensor_a.shape()
    if is_scalar(input_tensor_b):
        return Tensor(ttl.tensor.add_unary(input_tensor_a, input_tensor_b))
    elif not isinstance(input_tensor_b, ttl.tensor.Tensor):
        raise TypeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_b = input_tensor_b.shape()
    *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW))
    elif height_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H))
    elif width_b == 1:
        return Tensor(ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W))
    return Tensor(ttl.tensor.mul(input_shape_a, input_tensor_b))


ttl.tensor.Tensor.__matmul__ = matmul
ttl.tensor.Tensor.__add__ = add
ttl.tensor.Tensor.__sub__ = subtract
ttl.tensor.Tensor.__mul__ = mul


# Data Transformations
def reshape(input_tensor, shape):

    try:
        w, z, y, x = shape
        return Tensor(ttl.tensor.reshape(input_tensor._tensor, w, z, y, x))
    except:
        logger.warning("Given reshape operation could not be run on the TT device. Defaulting to torch implementation")
        torch_tensor = to_torch(input_tensor)
        torch_tensor = torch_tensor.reshape(shape=shape)
        return from_torch(torch_tensor, input_tensor.dtype)


def permute(input_tensor, order):

    try:
        return Tensor(ttl.tensor.permute(input_tensor._tensor, order))
    except:
        logger.warning("Given permute operation could not be run on the TT device. Defaulting to torch implementation")
        torch_tensor = to_torch(input_tensor)
        torch_tensor = torch_tensor.permute(order)
        return from_torch(torch_tensor, input_tensor.dtype)


def softmax(input_tensor, dim):
    import torch

    torch_tensor = to_torch(input_tensor)
    torch_tensor = torch.softmax(torch_tensor, dim=dim)
    return from_torch(torch_tensor, dtype=input_tensor.dtype)
