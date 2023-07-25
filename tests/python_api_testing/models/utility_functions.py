import math

import time
import torch
import numpy as np
from loguru import logger
from tests.python_api_testing.models.utility_functions_new import (
    is_close,
    get_oom_of_float,
    get_compile_cache_enabled,
    enable_compilation_reports,
    disable_compilation_reports,
    enable_memory_reports,
    disable_memory_reports,
    comp_allclose,
    comp_pcc,
    comp_allclose_and_pcc,
    Profiler,
    profiler,
    tt_to_torch_tensor,
    torch_to_tt_tensor,
    torch_to_tt_tensor_rm,
    unpad_from_zero,
    pad_by_zero,
    tt2torch_tensor,
    torch2tt_tensor,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache
)

import tt_lib as ttl
from tt_lib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
    channels_last,
    convert_weights_2d_matrix,
)
