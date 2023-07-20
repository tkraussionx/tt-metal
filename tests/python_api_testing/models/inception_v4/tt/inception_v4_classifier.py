import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_conv2d import (
    TtInterceptV4Conv2D,
)
from python_api_testing.models.inception_v4.inception_v4_mini_graphs import TtIdentity
from models.helper_funcs import Linear as TtLinear
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtClassifier(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        num_features,
        num_classes,
        pool_type="avg",
        use_conv=False,
    ):
        super().__init__()
        self.device = device
        base_address = f"last_linear"

        flatten_in_pool = not use_conv
        if not pool_type:
            assert (
                num_classes == 0 or use_conv
            ), "Pooling can only be disabled if classifier is also removed or conv classifier is used"
            flatten_in_pool = (
                False  # disable flattening if pooling is pass-through (no pooling)
            )

        if pool_type != "avg":
            raise NotImplementedError

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1) if flatten_in_pool else nn.Identity()

        if num_classes <= 0:
            self.fc = TtIdentity()  # pass-through (no classifier)
        elif use_conv:
            self.fc = TtInterceptV4Conv2D(
                state_dict, base_address, device, num_features, num_classes, 1
            )
        else:
            fc_weight = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.weight"], self.device
            )
            fc_bias_key = state_dict[f"{base_address}.bias"]
            if fc_bias_key in state_dict:
                fc_bias = torch_to_tt_tensor_rm(fc_bias_key, self.device)
            else:
                fc_bias = None

            self.fc = TtLinear(
                num_features,
                num_classes,
                weight=fc_weight,
                bias=fc_bias,
            )

    def forward(
        self, x: tt_lib.tensor.Tensor, pre_logits: bool = False
    ) -> tt_lib.tensor.Tensor:
        x = tt2torch_tensor(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = torch_to_tt_tensor_rm(x, self.device, put_on_device=False)
        return x if pre_logits else self.fc(x)
