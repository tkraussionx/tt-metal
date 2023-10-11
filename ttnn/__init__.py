import tt_lib as ttl

# Conversion
def to_torch(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.layout() != ttl.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(ttl.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()

def from_torch(
    torch_tensor,
    dtype,
):
    print(torch_tensor)
    print(torch_tensor.dtype)
    print(dtype)
    return ttl.tensor.Tensor(torch_tensor, dtype)


# Initializers
def zeros(*args, **kwargs):
    return ttl.operations.zeros(*args, **kwargs)


def random(*args, **kwargs):
    return ttl.operations.random(*args, **kwargs)


# Math Operations
def matmul(*args, **kwargs):
    return ttl.tensor.matmul(*args, **kwargs)


def add(*args, **kwargs):
    return ttl.tensor.add(*args, **kwargs)


def subtract(*args, **kwargs):
    return ttl.tensor.sub(*args, **kwargs)


def multiply(*args, **kwargs):
    return ttl.tensor.mul(*args, **kwargs)


ttl.tensor.Tensor.__matmul__ = matmul
ttl.tensor.Tensor.__add__ = add
ttl.tensor.Tensor.__sub__ = subtract
ttl.tensor.Tensor.__mul__ = multiply


# Data Transformations
def reshape(*args, **kwargs):
    input_tensor, shape = args
    w, z, y, x = shape
    return ttl.tensor.reshape(input_tensor, w, z, y, x, **kwargs)


def permute(*args, **kwargs):
    return ttl.tensor.permute(*args, **kwargs)

# Activations
def softmax(input_tensor, dim):
    import torch
    torch_tensor = to_torch(input_tensor)
    torch_tensor = torch.softmax(torch_tensor, dim=dim)
    return from_torch(torch_tensor, dtype=input_tensor.dtype())
