from loguru import logger

import tt_lib as ttl

from ttnn.tensor import Tensor, from_torch, to_torch


def open(device_id: int):
    try:
        device = ttl.device.CreateDevice(device_id)
        ttl.device.SetDefaultDevice(device)
        return device
    except RuntimeError as e:
        if str(e).startswith("Cannot re-initialize device"):
            ttl.device.CloseDevice(device)
            device = ttl.device.CreateDevice(device_id)
            ttl.device.SetDefaultDevice(device)
            return device
        else:
            raise


def close(device):
    try:
        ttl.device.CloseDevice(device)
    except:
        pass


def is_scalar(value):
    return isinstance(value, (int, float, complex))


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


def _reshape_to_4D(tensor):
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    num_missing_dims = 4 - len(tensor.shape)
    shape = ([1] * num_missing_dims) + tensor.shape
    return reshape(tensor, shape=shape)

# Math Operations
# Should the matmal autodetect if the tensor is on device?
#   * Should one type of operation be prefered over the other for optimizations?
def matmul(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor:
    """
    matmul(input_tensor_a, input_tensor_b) -> Tensor

    Returns the matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both arguments are 2-dimensional, the matrix-matrix product is returned in 4-dimensional.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension to both tensors until they become 4-dimensional
      and will return a 4 dimensional result.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned in 4 dimensions.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple.
      The non-matrix (i.e. batch) dimensions must be broadcastable.  For example, if :attr:`input_tensor_a` is a
      :math:`(j \times 1 \times n \times n)` tensor and :attr:`input_tensor_b` is a :math:`(k \times n \times n)`
      tensor, :attr:`out` will be a :math:`(j \times k \times n \times n)` tensor.

      Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
      are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
      :math:`(j \times 1 \times n \times m)` tensor and :attr:`input_tensor_b` is a :math:`(k \times m \times p)`
      tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
      matrix dimensions) are different. :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.


    .. note::

        The 1-dimensional dot product version of this function is not currently supported.

    Arguments:
        input_tensor_a (Tensor): the first tensor to be multiplied
        input_tensor_b (Tensor): the second tensor to be multiplied

    Example::

        >>> # vector x vector
        >>> tensor1 = ttnn.from_torch(torch.randn(3))
        >>> tensor2 = ttnn.from_torch(torch.randn(3))
        >>> ttnn.matmul(tensor1, tensor2).size()
        torch.Size([1, 1, 1, 3])
        >>> # matrix x vector
        >>> tensor1 = torch.randn(3, 4)
        >>> tensor2 = torch.randn(4)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([3])
        >>> # batched matrix x broadcasted vector
        >>> tensor1 = torch.randn(10, 3, 4)
        >>> tensor2 = torch.randn(4)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3])
        >>> # batched matrix x batched matrix
        >>> tensor1 = torch.randn(10, 3, 4)
        >>> tensor2 = torch.randn(10, 4, 5)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3, 5])
        >>> # batched matrix x broadcasted matrix
        >>> tensor1 = torch.randn(10, 3, 4)
        >>> tensor2 = torch.randn(4, 5)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3, 5])
    """
    if not isinstance(input_tensor_a, Tensor):
        raise RuntimeError("Expected first argument to be a tt_lib.tensor.Tensor")
    if not isinstance(input_tensor_b, Tensor):
        raise RuntimeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_a = input_tensor_a.shape
    input_shape_b = input_tensor_b.shape

    # The idea is to make the shapes "possibly" broadcastable.
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    input_tensor_b = _reshape_to_4D(input_tensor_b)

    *_, height_a, width_a = input_shape_a
    *rest_of_shape_b, height_b, width_b = input_shape_b

    # if height_a % 32 != 0 or width_a % 32 != 0:
    #     raise TypeError("The last two dimensions of the first tensor must be a multiple of 32")

    # if height_b % 32 != 0 or width_b % 32 != 0:
    #     raise TypeError("The last two dimensions of the second tensor must be a multiple of 32")

    if width_a != height_b and height_a != 1 and height_b != 1:
        raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")

    if height_b == 1 and width_b == 1:
        return Tensor(
            ttl.tensor.bcast(
                input_tensor_a._tensor, input_tensor_b._tensor, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW
            )
        )
    elif _shape_is_broadcastable(input_shape_a, input_shape_b):
        if all(x == 1 for x in rest_of_shape_b):
            if width_a != height_b:
                raise RuntimeError("The width of the first tensor must be equal to the height of the second tensor")
            return Tensor(ttl.tensor.matmul(input_tensor_a._tensor, input_tensor_b._tensor))
        else:
            return Tensor(ttl.tensor.bmm(input_tensor_a._tensor, input_tensor_b._tensor))
    else:
        raise RuntimeError("These tensors cannot be broadcasted")


def add(input_tensor_a: Tensor, input_tensor_b: Tensor, *, alpha=1) -> Tensor:
    """
    add(input_tensor_a, input_tensor_b, *, alpha=1) -> Tensor

    Adds :attr:`input_tensor_b`, scaled by :attr:`alpha`, to :attr:`input_tensor_a`.

    .. math::
        \mathrm{{input\_tensor\_a}}_i + \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (Tensor or Number): the tensor or number to add to :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> a = ttnn.from_torch(torch.tensor((1, 2)))
        >>> b = ttnn.from_torch(torch.tensor((0, 1)))
        >>> ttnn.add(a, b, alpha=2)
        tensor([1, 4])
    """

    original_shape = input_tensor_a.shape
    input_tensor_a = _reshape_to_4D(input_tensor_a)
    ttl_input_tensor_a = input_tensor_a._tensor

    if ttl_input_tensor_a.storage_type() != ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("input_tensor_a must be on device!")

    if is_scalar(input_tensor_b):
        ttl_input_tensor_b = input_tensor_b
    else:
        input_tensor_b = _reshape_to_4D(input_tensor_b)
        ttl_input_tensor_b = input_tensor_b._tensor
        if ttl_input_tensor_b.storage_type() != ttl.tensor.StorageType.DEVICE:
            raise RuntimeError("input_tensor_a must be on device!")

    if not isinstance(ttl_input_tensor_a, ttl.tensor.Tensor):
        raise TypeError("Expected first argument to be a tt_lib.tensor.Tensor")

    if is_scalar(ttl_input_tensor_b):
        output_tensor = Tensor(ttl.tensor.add_unary(ttl_input_tensor_a, ttl_input_tensor_b * alpha))
        return reshape(output_tensor, original_shape)
    elif not isinstance(ttl_input_tensor_b, ttl.tensor.Tensor):
        raise TypeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_b = ttl_input_tensor_b.shape()

    if alpha != 1:
        ttl_input_tensor_b = ttl.tensor.mul_unary(ttl_input_tensor_b, alpha)

    *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)
        )
    elif height_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
        )
    elif width_b == 1:
        output_tensor = Tensor(
            ttl.tensor.bcast(ttl_input_tensor_a, ttl_input_tensor_b, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W)
        )
    else:
        output_tensor = Tensor(ttl.tensor.add(ttl_input_tensor_a, ttl_input_tensor_b))

    output_tensor = reshape(output_tensor, original_shape)
    return output_tensor


def subtract(input_tensor_a: Tensor, input_tensor_b: Tensor, *, alpha=1) -> Tensor:
    """
    sub(input_tensor_a, input_tensor_b, *, alpha=1) -> Tensor

    Subtracts :attr:`input_tensor_b`, scaled by :attr:`alpha`, from :attr:`input_tensor_a`.

    .. math::
        \mathrm{{input\_tensor\_a}}_i - \mathrm{{alpha}} \\times \mathrm{{input\_tensor\_b}}_i

    Supports broadcasting.

    Args:
        * :attr:`input_tensor_a`
        * :attr:`input_tensor_b` (Tensor or Number): the tensor or number to subtract from :attr:`input_tensor_a`.

    Keyword args:
        :attr:`alpha` (Number): the multiplier for :attr:`input_tensor_b`.

    Example::

        >>> a = ttnn.from_torch(torch.tensor((1, 2)))
        >>> b = ttnn.from_torch(torch.tensor((0, 1)))
        >>> ttnn.sub(a, b, alpha=2)
        tensor([1, 0])
    """
    input_tensor_a = input_tensor_a._tensor if isinstance(input_tensor_a, Tensor) else input_tensor_a
    input_tensor_b = input_tensor_b._tensor if isinstance(input_tensor_b, Tensor) else input_tensor_b

    if not isinstance(input_tensor_a, ttl.tensor.Tensor):
        raise TypeError("Expected first argument to be a tt_lib.tensor.Tensor")

    if is_scalar(input_tensor_b):
        return Tensor(ttl.tensor.add_unary(input_tensor_a, input_tensor_b * alpha))
    elif not isinstance(input_tensor_b, ttl.tensor.Tensor):
        raise TypeError("Expected second argument to be a tt_lib.tensor.Tensor or a scalar")

    input_shape_b = input_tensor_b.shape()

    if alpha != 1:
        input_tensor_b = ttl.tensor.mul_unary(input_tensor_b, alpha)

    *_, height_b, width_b = input_shape_b

    if height_b == 1 and width_b == 1:
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW)
        )
    elif height_b == 1:
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H)
        )
    elif width_b == 1:
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W)
        )
    return Tensor(ttl.tensor.sub(input_tensor_a, input_tensor_b))


def multiply(input_tensor_a: Tensor, input_tensor_b: Tensor) -> Tensor:
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
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW)
        )
    elif height_b == 1:
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)
        )
    elif width_b == 1:
        return Tensor(
            ttl.tensor.bcast(input_tensor_a, input_tensor_b, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)
        )
    return Tensor(ttl.tensor.mul(input_shape_a, input_tensor_b))


sub = subtract
mul = multiply


Tensor.__matmul__ = matmul
Tensor.__add__ = add
Tensor.__sub__ = subtract
Tensor.__mul__ = multiply


# Data Transformations
def reshape(input_tensor: Tensor, shape) -> Tensor:
    ttl_input_tensor = input_tensor._tensor

    if ttl_input_tensor.layout() == ttl.tensor.Layout.ROW_MAJOR:
        return Tensor(ttl_input_tensor.reshape(shape))

    try:
        w, z, y, x = shape
        return Tensor(ttl.tensor.reshape(ttl_input_tensor, w, z, y, x))
    except:
        logger.warning("Given reshape operation could not be run on the TT device. Defaulting to torch implementation")
        torch_tensor = to_torch(input_tensor)
        torch_tensor = torch_tensor.reshape(shape=shape)
        return from_torch(torch_tensor, input_tensor.dtype)


def permute(input_tensor: Tensor, order) -> Tensor:
    try:
        return Tensor(ttl.tensor.permute(input_tensor._tensor, order))
    except:
        logger.warning("Given permute operation could not be run on the TT device. Defaulting to torch implementation")
        torch_tensor = to_torch(input_tensor)
        torch_tensor = torch_tensor.permute(order)
        return from_torch(torch_tensor, input_tensor.dtype)


def softmax(input_tensor: Tensor, dim) -> Tensor:
    import torch

    torch_tensor = to_torch(input_tensor)
    torch_tensor = torch.softmax(torch_tensor, dim=dim)
    return from_torch(torch_tensor, dtype=input_tensor.dtype)


__all__ = [
    "matmul",
    "add",
    "sub",
    "subtract",
    "mul",
    "multiply",
    "reshape",
    "permute",
    "softmax",
]
