import torch
import json
import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_device)

    return tt_tensor



def torch2tt_tensor_3(py_tensor: torch.Tensor, tt_device):
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
    )

    return tt_tensor


def torch2tt_tensor_2(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    return tt_tensor

def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def tt_const_tensor(value, shape, device):
    pytorch_const = torch.full(shape, value)
    tt_const = torch2tt_tensor(pytorch_const, device)
    return tt_const


def tt_linear(x, weight, bias=None):
    #weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x

def pad_input_tensor(tensor, value, multiple):
    len = tensor.shape[1]

    if len % multiple == 0:
        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def tt_matmul(t1, t2, device, on_torch=False):
    if on_torch:
        t1 = tt2torch_tensor(t1)
        t2 = tt2torch_tensor(t2)

        res = torch.matmul(t1, t2)
        return torch2tt_tensor(res, device)
    else:
        return tt_lib.tensor.bmm(t1, t2)


def tt_bmm(t1, t2, device, on_torch=False):
    if on_torch:
        return tt_matmul(t1, t2, device)
    else:
        return tt_lib.tensor.bmm(t1, t2)


def read_model_config(json_file):
    # read file
    with open(json_file, "r") as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    return obj

def pad_input_tensor(tensor, value, multiple):
    tensor = torch.transpose(tensor, 0, 1)
    len = tensor.shape[1]

    if len % multiple == 0:
        tensor = torch.transpose(tensor, 0, 1)

        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    tensor = torch.transpose(tensor, 0, 1)

    return tensor
