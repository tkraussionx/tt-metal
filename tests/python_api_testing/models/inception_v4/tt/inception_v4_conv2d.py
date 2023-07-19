import torch
import tt_lib
from loguru import logger
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.conv_on_device_utils_new import (
    run_conv_on_tt_device,
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)


class TtInterceptV4Conv2D(torch.nn.Module):
    def __init__(
        self,
        state_dict,
        base_address,
        device,
        c1,
        c2,
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        conv_on_device=False,
    ):
        super().__init__()

        self.device = device
        self.host = tt_lib.device.GetHost()
        self.conv_on_device = conv_on_device

        # Initialize conv2d
        self.conv_weight = state_dict[f"{base_address}.weight"]

        bias_key = f"{base_address}.bias"
        if bias_key in state_dict:
            self.conv_bias = state_dict[bias_key]
        else:
            self.conv_bias = None

        self.conv_params = [c2, c1, k, k, s, s, p, p, d, g]

        if self.conv_on_device and is_conv_supported_on_device(self.conv_params):
            self.conv_bias = self.conv_bias.unsqueeze(-1).unsqueeze(-1)
            logger.debug(f"Using TtConv for params {self.conv_params}")

            self.conv = run_conv_on_device_wrapper(
                self.conv_weight.reshape(-1).tolist(),
                self.conv_params,
                self.device,
                self.host,
                conv_bias=None,
            )

        else:
            self.conv_on_device = False
            logger.debug(f"Using fallback_ops.Conv2d for params {self.conv_params}")

            self.conv = fallback_ops.Conv2d(
                weights=self.conv_weight,
                biases=self.conv_bias,
                in_channels=c1,
                out_channels=c2,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=g,
                dilation=d,
                bias=self.conv_bias is not None,
            )

    def forward(self, x):
        if self.conv_on_device:
            x = tt2torch_tensor(x)
            x = self.conv(x)
            x = x + self.conv_bias
            x = torch2tt_tensor(
                x, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
            )
        else:
            x = self.conv(x)

        return x
