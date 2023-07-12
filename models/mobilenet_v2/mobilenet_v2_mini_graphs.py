from typing import Any
import tt_lib
import torch
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)
from loguru import logger


class TtIdentity(torch.nn.Module):
    """
    Implementation of torch.nn.Identity op

    A placeholder identity operator that is argument-insensitive.
    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(self, input: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        return input
