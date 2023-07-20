import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from python_api_testing.models.inception_v4.tt.inception_v4_mixed3a import (
    TtMixed3a,
)
from python_api_testing.models.inception_v4.tt.inception_v4_mixed4a import (
    TtMixed4a,
)
from python_api_testing.models.inception_v4.tt.inception_v4_mixed5a import (
    TtMixed5a,
)
from python_api_testing.models.inception_v4.tt.inception_v4_inceptiona import (
    TtInceptionA,
)
from python_api_testing.models.inception_v4.tt.inception_v4_inceptionb import (
    TtInceptionB,
)
from python_api_testing.models.inception_v4.tt.inception_v4_inceptionc import (
    TtInceptionC,
)
from python_api_testing.models.inception_v4.tt.inception_v4_reductiona import (
    TtReductionA,
)
from python_api_testing.models.inception_v4.tt.inception_v4_reductionb import (
    TtReductionB,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm

from python_api_testing.models.inception_v4.tt.inception_v4_classifier import (
    TtClassifier,
)
import timm


class TtInceptionV4(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        drop_rate=0.0,
        global_pool="avg",
    ):
        super().__init__()
        self.device = device
        assert output_stride == 32

        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536

        self.features = nn.Sequential(
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.0",
                in_planes=in_chans,
                out_planes=32,
                kernel_size=3,
                stride=2,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.1",
                in_planes=32,
                out_planes=32,
                kernel_size=3,
                stride=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.2",
                in_planes=32,
                out_planes=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            TtMixed3a(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.3",
            ),
            TtMixed4a(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.4",
            ),
            TtMixed5a(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.5",
            ),
            TtInceptionA(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.6",
            ),
            TtInceptionA(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.7",
            ),
            TtInceptionA(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.8",
            ),
            TtInceptionA(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.9",
            ),
            TtReductionA(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.10",
            ),  # Mixed6a
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.11",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.12",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.13",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.14",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.15",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.16",
            ),
            TtInceptionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.17",
            ),
            TtReductionB(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.18",
            ),  # Mixed7a
            TtInceptionC(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.19",
            ),
            TtInceptionC(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.20",
            ),
            TtInceptionC(
                device=self.device,
                state_dict=state_dict,
                base_address=f"features.21",
            ),
        )
        self.feature_info = [
            dict(num_chs=64, reduction=2, module="features.2"),
            dict(num_chs=160, reduction=4, module="features.3"),
            dict(num_chs=384, reduction=8, module="features.9"),
            dict(num_chs=1024, reduction=16, module="features.17"),
            dict(num_chs=1536, reduction=32, module="features.21"),
        ]
        self.classifier = TtClassifier(
            self.device,
            state_dict,
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
        )

    def get_classifier(self):
        return self.classifier

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.classifier(x, pre_logits)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _inception_v4(
    device, state_dict, num_classes, in_chans, global_pool="avg"
) -> TtInceptionV4:
    return TtInceptionV4(
        device,
        state_dict,
        num_classes=num_classes,
        in_chans=in_chans,
        global_pool="avg",
    )


def inception_v4(device) -> TtInceptionV4:
    reference_model = timm.create_model("inception_v4", pretrained=True)
    state_dict = reference_model.state_dict()
    num_classes = reference_model.num_classes
    in_chans = reference_model.features[0].conv.in_channels
    return _inception_v4(device, state_dict, num_classes, in_chans)
