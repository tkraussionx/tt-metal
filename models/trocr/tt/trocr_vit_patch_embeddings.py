import torch.nn as nn
import collections

import tt_lib
from tt_lib import fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


class TtViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, base_address, state_dict, device, host):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.device = device
        self.host = host

        conv_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.projection.weight"],
            self.device,
            put_on_device=False,
        )
        conv_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.projection.bias"],
            self.device,
            put_on_device=False,
        )
        self.projection = fallback_ops.Conv2d(
            weights=conv_weight,
            biases=conv_bias,
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            padding_mode="zeros",
        )

    def forward(
        self, pixel_values: tt_lib.tensor.Tensor, interpolate_pos_encoding: bool = False
    ) -> tt_lib.tensor.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape()
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(pixel_values)
        embeddings = tt_to_torch_tensor(embeddings, self.host)
        embeddings = embeddings.flatten(2)
        embeddings = torch_to_tt_tensor_rm(embeddings, self.host)
        embeddings = tt_lib.tensor.transpose(embeddings)
        return embeddings
