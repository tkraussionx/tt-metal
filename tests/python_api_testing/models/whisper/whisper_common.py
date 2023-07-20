import torch
import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    if size[-1] % 2 != 0:
        tt_device = tt_lib.device.GetHost()

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)
    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).to(torch.float).reshape(tt_output.shape())
    return py_output


def linear(x, weight, bias=None):
    weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x


def create_padded_tensor(
    input_tensors_shape,
    input_tensor,
    output_tensor_shape,
    pad_value,
    device,
    input_tensor_start=[0, 0, 0, 0],
):
    while len(input_tensors_shape) < 4:
        input_tensors_shape.insert(0, 1)

    if isinstance(input_tensor, tt_lib.tensor.Tensor):
        torch_tensor = torch.Tensor(input_tensor.data()).reshape(input_tensor.shape())
    else:
        torch_tensor = input_tensor

    # Create tensor on host
    a = tt_lib.tensor.Tensor(
        torch_tensor.reshape(-1).tolist(),
        input_tensors_shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )
    # Pad inputs on host
    a_pad = a.pad(output_tensor_shape, input_tensor_start, pad_value)

    # a_pt = torch.Tensor(a_pad.data()).reshape(*output_tensor_shape)
    a_dev = a_pad.to(tt_lib.tensor.Layout.TILE).to(device)

    return a_dev


def create_unpadded_tensor(
    ttm_tensor, input_tensors_shape, input_tensor_start=[0, 0, 0, 0]
):
    output_tensor_start = input_tensor_start
    output_tensor_end = tuple(
        input_tensor_start[i] + input_tensors_shape[i] - 1
        for i in range(len(input_tensors_shape))
    )
    ttm_tensor = (
        ttm_tensor.to(tt_lib.device.GetHost())
        .to(tt_lib.tensor.Layout.ROW_MAJOR)
        .unpad(output_tensor_start, output_tensor_end)
    )

    return ttm_tensor
