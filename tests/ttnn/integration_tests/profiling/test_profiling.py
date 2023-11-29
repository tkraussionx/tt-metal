# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import os
from ttnn.profile import append_torch_results_to_file, set_profiler_location
from tests.ttnn.utils_for_testing import update_process_id


def test_torch_profiling(torch_performance_csv, use_torch_profiling):
    torch_a = torch.rand((1, 1, 3, 4), dtype=torch.bfloat16)
    torch_a = torch_a.expand(10, 10, 3, 4)
    append_torch_results_to_file(torch_performance_csv)
    lines = []
    with open(torch_performance_csv, "r") as temp_file:
        lines = temp_file.readlines()
    assert len(lines) == 8


def list_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def test_tt_profiling(tt_profile_location, use_tt_profiling):
    update_process_id()
    torch_a = torch.rand((1, 1, 3, 4), dtype=torch.bfloat16)
    torch_a = torch_a.expand(10, 10, 3, 4)
    file_paths = list_files(tt_profile_location)
    # with open(torch_performance_csv, "r") as temp_file:
    #     lines = temp_file.readlines()
    assert len(file_paths) == 1
