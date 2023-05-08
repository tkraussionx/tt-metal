from .. import tensor as ttl_tensor, device as ttl_device
import torch
from functools import wraps

def convert_tt_tensors_wrapper(func):
    host = ttl_device.GetHost()

    @wraps(func)
    def wrap(*args, **kwargs):
        output_format = {}

        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, ttl_tensor.Tensor):
                if not output_format:
                    if args[i].on_host():
                        output_format["device"] = host
                    else:
                        output_format["device"] = args[i].device()
                    output_format["layout"] = args[i].layout()
                    output_format["dtype"] = args[i].dtype()
                # Convert to PT Tensor
                new_args.append(
                    torch.Tensor(
                        arg.to(host).to(ttl_tensor.Layout.ROW_MAJOR).data()
                    ).reshape(arg.shape())
                )
            else:
                new_args.append(args[i])

        for key, value in kwargs.items():
            if isinstance(value, ttl_tensor.Tensor):
                if not output_format:
                    if value.on_host():
                        output_format["device"] = host
                    else:
                        output_format["device"] = value.device()
                    output_format["layout"] = value.layout()
                    output_format["dtype"] = value.dtype()
                # Convert to PT Tensor
                kwargs[key] = torch.Tensor(
                    value.to(host).to(ttl_tensor.Layout.ROW_MAJOR).data()
                ).reshape(value.shape())

        # Set default output format
        if not output_format:
            output_format = {"device": host, "layout": ttl_tensor.Layout.ROW_MAJOR, "dtype": ttl_tensor.DataType.BFLOAT16}

        outputs = func(*new_args, **kwargs)

        # CONVERT TO TT TENSOR
        if outputs is None:
            return outputs
        elif isinstance(outputs, torch.Tensor):
            output = ttl_tensor.Tensor(
                outputs.reshape(-1).tolist(),
                outputs.shape,
                output_format["dtype"],
                ttl_tensor.Layout.ROW_MAJOR,
            )
            if output_format["layout"] == ttl_tensor.Layout.TILE:
                if (
                    output.shape()[2] % 32 == 0 and output.shape()[3] % 32 == 0
                ):  # Restore tile layout only if legal or else leave as RM
                    output = output.to(ttl_tensor.Layout.TILE)
            else:
                output = output.to(output_format["layout"])
            output = output.to(output_format["device"])
            return output

        new_outputs = []
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            for i, output in enumerate(outputs):
                # Convert to TT Tensor
                if isinstance(output, torch.Tensor):
                    output = ttl_tensor.Tensor(
                        output.reshape(-1).tolist(),
                        output.shape,
                        output_format["dtype"],
                        ttl_tensor.Layout.ROW_MAJOR,
                    )
                    if output_format["layout"] == ttl_tensor.Layout.TILE:
                        if (
                            output.shape()[2] % 32 == 0 and output.shape()[3] % 32 == 0
                        ):  # Restore tile layout only if legal or else leave as RM
                            output = output.to(ttl_tensor.Layout.TILE)
                    else:
                        output = output.to(output_format["layout"])
                    output = output.to(output_format["device"])
                    new_outputs.append(output)
        return new_outputs

    return wrap
