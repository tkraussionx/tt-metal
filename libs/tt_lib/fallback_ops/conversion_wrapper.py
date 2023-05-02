from .. import tensor as ttl_tensor, device as ttl_device
import torch


def convert_tt_tensors_wrapper(func):
    host = ttl_device.GetHost()

    def wrap(*args, **kwargs):
        device = None
        layout = None
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, ttl_tensor.Tensor):
                if device is None and not args[i].on_host():
                    device = args[i].device()
                if layout is None:
                    layout = args[i].layout()
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
                if device is None and not value.on_host():
                    device = value.device()
                # Convert to PT Tensor
                kwargs[key] = torch.Tensor(
                    value.to(host).to(ttl_tensor.Layout.ROW_MAJOR).data()
                ).reshape(value.shape())

        outputs = func(*new_args, **kwargs)

        # CONVERT TO TT TENSOR
        if outputs is None:
            return outputs
        elif isinstance(outputs, torch.Tensor):
            output = ttl_tensor.Tensor(
                outputs.reshape(-1).tolist(),
                outputs.shape,
                ttl_tensor.DataType.BFLOAT16,
                ttl_tensor.Layout.ROW_MAJOR,
            )
            if layout == ttl_tensor.Layout.TILE:
                if (
                    output.shape()[2] % 32 == 0 and output.shape()[3] % 32 == 0
                ):  # Restore tile layout only if legal or else leave as RM
                    output = output.to(ttl_tensor.Layout.TILE)
            else:
                output = output.to(layout)
            if device is not None:
                output = output.to(device)
            return output

        new_outputs = []
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            for i, output in enumerate(outputs):
                # Convert to TT Tensor
                if isinstance(output, torch.Tensor):
                    output = ttl_tensor.Tensor(
                        output.reshape(-1).tolist(),
                        output.shape,
                        ttl_tensor.DataType.BFLOAT16,
                        ttl_tensor.Layout.ROW_MAJOR,
                    )
                    if layout == ttl_tensor.Layout.TILE:
                        if (
                            output.shape()[2] % 32 == 0 and output.shape()[3] % 32 == 0
                        ):  # Restore tile layout only if legal or else leave as RM
                            output = output.to(ttl_tensor.Layout.TILE)
                    else:
                        output = output.to(layout)
                    if device is not None:
                        output = output.to(device)
                    new_outputs.append(output)
        return new_outputs

    return wrap
