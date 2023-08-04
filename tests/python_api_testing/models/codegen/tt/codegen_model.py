import torch
from torch.nn import functional as F

import tt_lib
from python_api_testing.models.helper_funcs import Linear
import python_api_testing.models.codegen.tt.codegen_gelu as codegen_gelu
import python_api_testing.models.codegen.tt.codegen_block as codegen_block
from tt_lib.fallback_ops import fallback_ops


from torch import nn

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import CodeGenConfig

class TtCodeGenModel(torch.nn.Module):
    def __init__(self, config: CodeGenConfig(), state_dict, device):
        super().__init__()


        self.config = config
        self.embed_dim = config.n_embd
        self.vocab_size = 50400
        self.n_layer = 19
        self.hidden_size = 1024

        print('CONFIG')
        print(config)
        print('PARAME')
        print(self.vocab_size)
        print(self.embed_dim)


        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

        blocks = []

        for i in range(self.n_layer):
            block = codegen_block.TtCodeGenBlock(i, config, state_dict, device)
            blocks.append(block)

        self.h = torch.nn.ModuleList(blocks)


        self.beta = torch_to_tt_tensor_rm(
            state_dict["ln_f.bias"], device
        )

        self.gamma = torch_to_tt_tensor_rm(
            state_dict["ln_f.weight"], device
        )

        self.ln_f = fallback_ops.LayerNorm(
            self.gamma,
            self.beta,
            eps=config.layer_norm_epsilon,
            normalized_shape=self.hidden_size,
        )

        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)


    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = (
                head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = head_mask.to(
            dtype=self.dtype
        )  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (Tensor with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            Tensor with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings


    def forward(
        self,
        device,
        input_ids,
        past_key_values= None,
        attention_mask = None,
        token_type_ids = None,
        position_ids  = None,
        head_mask = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape()

            input_ids = tt_lib.fallback_ops.reshape(input_ids, 1, 1, -1, input_shape[-1])

            input_ids_shape = input_ids.shape()

            batch_size = input_ids_shape[0]


        elif inputs_embeds is not None:
            input_shape_2 = inputs_embeds.shape()
            input_shape_2 = input_shape_2[:-1]
            batch_size = inputs_shape_2.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        #device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:

            input_shape_3 = input_ids.shape()

            token_type_ids = tt_lib.fallback_ops.reshape(token_type_ids, 1, 1, -1, input_shape_3[-1])

        if position_ids is not None:
            input_shape_4 = input_ids.shape()

            position_ids = tt_lib.fallback_ops.reshape(position_ids, 1, 1, -1, input_shape_4[-1])


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            # Problem???
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            input_shape_5 = input_ids.shape()

            position_ids = torch.arange(past_length, input_shape_5[-1] + past_length)

            position_ids = tt_lib.fallback_ops.reshape(position_ids, 1, 1, -1, input_shape_5[-1])

            position_ids = tt2torch_tensor(position_ids)

            position_ids = position_ids.unsqueeze(0)
            position_ids = torch2tt_tensor(position_ids, device)

            position_ids = tt_lib.fallback_ops.reshape(position_ids, 1, 1, -1, input_shape_5[-1])


        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")



            attention_mask = tt_lib.fallback_ops.reshape(attention_mask, 1, 1, batch_size, -1)


            slice_list_attention_mask = [slice(0, attention_mask_shape[0]), slice(None), slice(None), slice(0, attention_mask_shape[3]) ]

            attention_mask = fallback_ops.tensor_slice(attention_mask, slice_list_attention_mask)


            tt_const_1 = fallback_ops.full(attention_maskshape(), 1.0)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

            attention_mask = tt_lib.tensor.sub(tt_const_1, attention_mask_shape)


        #pt_head_mask = tt2torch_tensor(head_mask)

        #pt_head_mask = self.get_head_mask(pt_head_mask, self.config.n_layer)


        if inputs_embeds is None:
            pt_input_ids = tt2torch_tensor(input_ids)
            pt_inputs_embeds = self.wte(pt_input_ids)
            input_embeds = torch2tt_tensor(pt_inputs_embeds, device)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            pt_token_type_ids = tt2torch_tensor(token_type_ids)

            pt_token_type_embeds = self.wte(pt_token_type_ids)

            token_type_embeds = torch2tt_tensor(pt_token_type_embeds, device)

            hidden_states = tt_lib.tensor.add(hidden_states, token_type_embeds)

        pt_hidden_states = tt2torch_tensor(hidden_states)

        pt_hidden_states = self.drop(pt_hidden_states)

        hidden_states = torch2tt_tensor(pt_hidden_states, device)

        input_shape_6 = input_ids.shape()
        hidden_states_size = hidden_states.shape()

        output_shape = input_shape_6 + (hidden_states_size[-1],)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        outputs_shapes = outputs.shape()

        hidden_states = tt_lib.fallback_ops.reshape(hidden_states, outputs_shape[0], outputs_shape[1], outputs_shape[2], outputs_shape[3])
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
