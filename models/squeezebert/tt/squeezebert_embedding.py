from typing import Optional
import torch
import torch.nn as nn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)

import tt_lib


class TtSqueezeBert_Embeddings(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()
        self.config = config
        self.base_address = base_address
        self.device = device

        self.word_embedding_weight = state_dict[
            f"{self.base_address}.word_embeddings.weight"
        ]

        self.word_embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_size,
            padding_idx=self.config.pad_token_id,
            _weight=self.word_embedding_weight,
        )

        self.position_embedding_weight = state_dict[
            f"{self.base_address}.position_embeddings.weight"
        ]
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.config.max_position_embeddings,
            embedding_dim=self.config.embedding_size,
            _weight=self.position_embedding_weight,
        )

        self.token_type_embedding_weight = state_dict[
            f"{self.base_address}.token_type_embeddings.weight"
        ]

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=self.config.type_vocab_size,
            embedding_dim=self.config.embedding_size,
            _weight=self.token_type_embedding_weight,
        )

        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.LayerNorm.weight"], self.device
        )
        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.LayerNorm.bias"], self.device
        )

        self.LayerNorm = tt_lib.tensor.layernorm

        self.register_buffer(
            "position_ids",
            torch.arange(self.config.max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
    ) -> tt_lib.tensor.Tensor:
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.shape()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        inputs_embeds = torch_to_tt_tensor_rm(
            inputs_embeds, self.device, put_on_device=True
        )
        position_embeddings = torch_to_tt_tensor_rm(
            position_embeddings, self.device, put_on_device=True
        )
        token_type_embeddings = torch_to_tt_tensor_rm(
            token_type_embeddings, self.device, put_on_device=True
        )

        embeddings = tt_lib.tensor.add(
            inputs_embeds, tt_lib.tensor.add(position_embeddings, token_type_embeddings)
        )

        embeddings = self.LayerNorm(
            embeddings, eps=self.config.layer_norm_eps, gamma=self.gamma, beta=self.beta
        )

        return embeddings
