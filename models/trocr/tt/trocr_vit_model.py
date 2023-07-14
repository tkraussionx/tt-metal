import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.trocr.tt.trocr_vit_embeddings import TtViTEmbeddings
from models.trocr.tt.trocr_vit_patch_embeddings import TtViTPatchEmbeddings
from models.trocr.tt.trocr_vit_encoder import TtViTEncoder
from models.trocr.tt.trocr_vit_pooler import TtViTPooler

import tt_lib
from tt_lib import fallback_ops


class TtViTModel(nn.Module):
    def __init__(
        self,
        config: TtViTConfig,
        base_address: str,
        state_dict: Dict,
        device,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
        host=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.host = host
        self.embeddings = TtViTEmbeddings(
            config,
            use_mask_token=use_mask_token,
            base_address=f"{base_address}.embeddings",
            state_dict=state_dict,
            host=self.host,
        )
        self.encoder = TtViTEncoder(
            config,
            base_address=f"{base_address}.encoder",
            state_dict=state_dict,
            device=device,
            host=self.host,
        )

        wln = state_dict[f"{base_address}.layernorm.weight"]
        bln = state_dict[f"{base_address}.layernorm.bias"]
        self.layernorm = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            weights=wln,
            biases=bln,
        )
        self.pooler = TtViTPooler(
            config,
            base_address=f"{base_address}.pooler",
            state_dict=state_dict,
            device=device,
            host=self.host,
        )

        # Initialize weights and apply final processing

    def get_input_embeddings(self) -> TtViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[tt_lib.tensor.Tensor] = None,
        bool_masked_pos: Optional[tt_lib.tensor.Tensor] = None,  # torch.booltensor
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, tt_lib.tensor.Tensor]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output)

        head_outputs = (
            (sequence_output, pooled_output)
            if pooled_output is not None
            else (sequence_output,)
        )
        head_outputs = head_outputs + encoder_outputs[1:]
        return head_outputs
