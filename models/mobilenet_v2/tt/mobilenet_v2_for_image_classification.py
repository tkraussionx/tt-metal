import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from typing import Optional, Union
from transformers import MobileNetV2Config
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from models.mobilenet_v2.tt.mobilenet_v2_model import (
    TtMobileNetV2Model,
)
from models.mobilenet_v2.mobilenet_v2_utils import (
    make_divisible,
    apply_depth_multiplier,
)
from models.helper_funcs import Linear as linear
from models.mobilenet_v2.mobilenet_v2_mini_graphs import TtIdentity
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)
from dataclasses import dataclass


@dataclass
class TtImageClassifierOutputWithNoAttention:
    loss: tt_lib.tensor.Tensor = None
    logits: tt_lib.tensor.Tensor = None
    hidden_states: tt_lib.tensor.Tensor = None


class TtMobileNetV2ForImageClassification(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        config: MobileNetV2Config,
    ):
        """
        MobileNetV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
        ImageNet.
        """
        super().__init__()

        self.device = device
        self.config = config

        self.num_labels = config.num_labels

        self.mobilenet_v2 = TtMobileNetV2Model(
            base_address="mobilenet_v2",
            state_dict=state_dict,
            config=config,
            device=device,
        )

        last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.conv_params[0]

        # Classifier head
        # No dropout used for inference
        # self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)

        classifier_weight = torch_to_tt_tensor_rm(
            state_dict[f"classifier.weight"], self.device
        )
        classifier_bias_key = state_dict[f"classifier.bias"]
        if classifier_bias_key in state_dict:
            classifier_bias = torch_to_tt_tensor_rm(classifier_bias_key, self.device)
        else:
            classifier_bias = None

        self.classifier = (
            linear(
                last_hidden_size,
                config.num_labels,
                weight=classifier_weight,
                bias=classifier_bias,
            )
            if config.num_labels > 0
            else TtIdentity()
        )

    def forward(
        self,
        pixel_values: Optional[tt_lib.tensor.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TtImageClassifierOutputWithNoAttention]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mobilenet_v2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        labels = None  # TODO: when implementing training. Not used for now
        if labels is not None:
            raise NotImplementedError
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
