import torch
import torch.nn as nn
from typing import Dict, Optional
import math
from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.trocr.tt.trocr_vit_patch_embeddings import TtViTPatchEmbeddings

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

import tt_lib
from tt_lib import fallback_ops


class TtViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(
        self,
        config: TtViTConfig,
        base_address: str,
        state_dict: Dict,
        use_mask_token: bool = False,
        device=None,
        host=None,
    ) -> None:
        super().__init__()
        self.host = host
        self.device = device
        self.cls_token = nn.Parameter(state_dict[f"{base_address}.cls_token"])

        self.mask_token = (
            nn.Parameter(tt_lib.tensor.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = TtViTPatchEmbeddings(
            config, f"{base_address}.patch_embeddings", state_dict, device, host
        )
        self.position_embeddings = nn.Parameter(
            state_dict[f"{base_address}.position_embeddings"]
        )
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: tt_lib.tensor.Tensor, height: int, width: int
    ) -> tt_lib.tensor.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = fallback_ops.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )

        patch_pos_embed = torch_to_tt_tensor_rm(patch_pos_embed, self.host)
        patch_pos_embed = fallback_ops.reshape(patch_pos_embed, 1, -1, dim)
        patch_pos_embed = tt_lib.tensor.permute(patch_pos_embed, 0, 2, 3, 1)
        patch_pos_embed = tt_to_torch_tensor(patch_pos_embed, self.host)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: tt_lib.tensor.Tensor,
        bool_masked_pos: Optional[tt_lib.tensor.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> tt_lib.tensor.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape()
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = tt_to_torch_tensor(embeddings, self.host)
        embeddings = torch.concat((cls_tokens, embeddings.squeeze(0)), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = torch_to_tt_tensor_rm(embeddings, self.host)
        return embeddings
