# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from torch.autograd import profiler
from functools import wraps
import tt_lib as ttl

from models.utility_functions import Profiler as TTProfiler

torch_profile_results = []
original_torch_functions = []
original_tt_functions = []


def profile_with_torch(method):
    @wraps(method)
    def profiled_method(*args, **kwargs):
        with profiler.profile(record_shapes=True, profile_memory=True) as prof:
            result = method(*args, **kwargs)
        torch_profile_results.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return result

    return profiled_method


def setup_torch_profiling():
    global original_torch_functions
    original_torch_functions.append(torch.arange)
    torch.arange = profile_with_torch(torch.arange)
    original_torch_functions.append(torch.Tensor.expand)
    torch.Tensor.expand = profile_with_torch(torch.Tensor.expand)


def teardown_torch_profiling():
    global original_torch_functions
    torch.arange = original_torch_functions[0]
    torch.Tensor.expand = original_torch_functions[1]
    original_torch_functions.clear()


def append_torch_results_to_file(filename):
    global torch_profile_results
    with open(filename, "a") as file:
        for result in torch_profile_results:
            file.write(result + "\n")
    torch_profile_results = []


def profile_with_tt(method):
    bar = ttl.tensor.decorate_external_operation(method, function_name=f"{method.__name__}")

    def profiled_method(*args, **kwargs):
        return bar(*args, **kwargs)

    return profiled_method


def setup_tt_profiling():
    global original_tt_functions
    original_tt_functions.append(torch.arange)
    torch.arange = profile_with_tt(torch.arange)
    original_tt_functions.append(torch.Tensor.expand)
    torch.Tensor.expand = profile_with_tt(torch.Tensor.expand)


def teardown_tt_profiling():
    global original_tt_functions
    torch.arange = original_tt_functions[0]
    torch.Tensor.expand = original_tt_functions[1]
    original_tt_functions.clear()


def set_profiler_location(directory):
    ttl.profiler.set_profiler_location(directory)
