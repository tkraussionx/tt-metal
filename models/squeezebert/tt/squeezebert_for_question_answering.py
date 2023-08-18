from typing import Optional, Tuple, Union
import torch.nn as nn
from dataclasses import dataclass

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib
from dataclasses import dataclass
from models.squeezebert.tt.squeezebert_model import TtSqueezeBertModel
from models.helper_funcs import Linear as TtLinear


@dataclass
class TtQuestionAnsweringModelOutput:
    loss: Optional[tt_lib.tensor.Tensor] = None
    start_logits: tt_lib.tensor.Tensor = None
    end_logits: tt_lib.tensor.Tensor = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtSqueezeBertForQuestionAnswering(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()
        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device
        self.num_labels = self.config.num_labels

        self.transformer = TtSqueezeBertModel(
            config,
            state_dict=self.state_dict,
            base_address=f"transformer",
            device=self.device,
        )
        self.qa_weight = torch_to_tt_tensor_rm(
            state_dict[f"qa_outputs.weight"], self.device
        )
        self.qa_bias = torch_to_tt_tensor_rm(
            state_dict[f"qa_outputs.bias"], self.device
        )
        self.qa_outputs = TtLinear(
            self.qa_weight.shape()[-1],
            self.qa_weight.shape()[-2],
            self.qa_weight,
            self.qa_bias,
        )

    def forward(
        self,
        input_ids: Optional[tt_lib.tensor.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        token_type_ids: Optional[tt_lib.tensor.Tensor] = None,
        position_ids: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        start_positions: Optional[tt_lib.tensor.Tensor] = None,
        end_positions: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtQuestionAnsweringModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        total_loss = None
        logits = tt_to_torch_tensor(logits)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = torch_to_tt_tensor_rm(
            start_logits.squeeze(-1), self.device, put_on_device=False
        )
        end_logits = torch_to_tt_tensor_rm(
            end_logits.squeeze(-1), self.device, put_on_device=False
        )

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TtQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
