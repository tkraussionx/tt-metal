# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.conv2d_common import run_short, mesh_device_fixture

parameters = {
    "short_sweep_suite": {
        "input_specs": [
            # Contains following params
            # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_x, stride_y, pad_x, pad_y, groups, bias, dilation]
            [1, 32, 1, 28, 28, 3, 3, 1, 1, 0, 0, 1, True, 1],
            [1, 100, 100, 14, 14, 3, 3, 1, 1, 1, 1, 100, False, 1],
            [1, 1008, 1008, 14, 14, 3, 3, 2, 2, 1, 1, 21, False, 1],
            [1, 1008, 1008, 7, 7, 3, 3, 1, 1, 1, 1, 21, False, 1],
            [1, 1024, 1024, 10, 10, 3, 3, 1, 1, 1, 1, 1024, False, 1],
            [1, 256, 1024, 128, 128, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1024, True, 1],
            [1, 1024, 1024, 19, 19, 3, 3, 2, 2, 1, 1, 1024, False, 1],
            [1, 1024, 1024, 19, 19, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 1024, 1024, 7, 7, 3, 3, 1, 1, 1, 1, 1024, False, 1],
            [1, 2048, 1024, 7, 7, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 104, 104, 28, 28, 3, 3, 1, 1, 1, 1, 13, False, 1],
            [1, 104, 104, 56, 56, 3, 3, 2, 2, 1, 1, 13, False, 1],
            [1, 1056, 1056, 48, 48, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 1056, 1056, 96, 96, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 112, 112, 14, 14, 5, 5, 2, 2, 2, 2, 112, False, 1],
            [1, 1152, 1152, 7, 7, 3, 3, 1, 1, 1, 1, 1152, False, 1],
            [1, 1152, 1152, 7, 7, 5, 5, 1, 1, 2, 2, 1152, False, 1],
            [1, 1152, 1152, 8, 8, 3, 3, 1, 1, 1, 1, 1152, False, 1],
            [1, 1152, 1152, 8, 8, 5, 5, 1, 1, 2, 2, 1152, False, 1],
            [1, 12, 12, 56, 56, 3, 3, 1, 1, 1, 1, 12, False, 1],
            [1, 120, 120, 14, 14, 1, 5, 1, 1, 0, 2, 120, False, 1],
            [1, 120, 120, 14, 14, 5, 1, 1, 1, 2, 0, 120, False, 1],
            [1, 120, 120, 14, 14, 5, 5, 1, 1, 2, 2, 120, False, 1],
            [1, 120, 120, 28, 28, 3, 3, 1, 1, 1, 1, 120, False, 1],
            [1, 120, 120, 28, 28, 5, 5, 1, 1, 2, 2, 120, False, 1],
            [1, 120, 120, 28, 28, 3, 3, 1, 1, 1, 1, 5, False, 1],
            [1, 120, 120, 40, 40, 5, 5, 1, 1, 2, 2, 120, False, 1],
            [1, 120, 120, 56, 56, 3, 3, 2, 2, 1, 1, 5, False, 1],
            [1, 1232, 1232, 14, 14, 3, 3, 1, 1, 1, 1, 11, False, 1],
            [1, 1232, 1232, 28, 28, 3, 3, 2, 2, 1, 1, 11, False, 1],
            [1, 1248, 1248, 9, 9, 3, 3, 1, 1, 1, 1, 1248, False, 1],
            [1, 1248, 1248, 9, 9, 5, 5, 1, 1, 2, 2, 1248, False, 1],
            [1, 128, 128, 1, 1, 3, 3, 1, 1, 1, 1, 128, False, 1],
            [1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 128, True, 1],
            [1, 128, 128, 150, 150, 3, 3, 1, 1, 1, 1, 128, False, 1],
            [1, 128, 128, 180, 320, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 128, 128, 200, 272, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, 8, False, 1],
            [1, 64, 128, 30, 40, 3, 3, 1, 1, 1, 1, 1, True, 1],
            [1, 128, 128, 5, 5, 3, 3, 2, 2, 1, 1, 128, False, 1],
            [1, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 128, False, 1],
            [1, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 8, False, 1],
            [1, 128, 128, 56, 56, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 256, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 32, 128, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 128, 128, 60, 80, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 1280, 1280, 30, 40, 3, 3, 1, 1, 1, 1, 1280, True, 1],
            [1, 1344, 1344, 14, 14, 3, 3, 1, 1, 1, 1, 8, False, 1],
            [1, 1344, 1344, 28, 28, 3, 3, 2, 2, 1, 1, 8, False, 1],
            [1, 1392, 1392, 10, 10, 3, 3, 1, 1, 1, 1, 1392, False, 1],
            [1, 1392, 1392, 10, 10, 5, 5, 1, 1, 2, 2, 1392, False, 1],
            [1, 1392, 1392, 14, 14, 3, 3, 1, 1, 1, 1, 6, False, 1],
            [1, 1392, 1392, 28, 28, 3, 3, 2, 2, 1, 1, 6, False, 1],
            [1, 144, 144, 14, 14, 5, 5, 1, 1, 2, 2, 144, False, 1],
            [1, 144, 144, 151, 151, 3, 3, 2, 2, 0, 0, 144, False, 1],
            [1, 144, 144, 191, 191, 3, 3, 2, 2, 0, 0, 144, False, 1],
            [1, 144, 144, 28, 28, 3, 3, 1, 1, 1, 1, 9, False, 1],
            [1, 144, 144, 56, 56, 3, 3, 1, 1, 1, 1, 144, False, 1],
            [1, 144, 144, 56, 56, 3, 3, 2, 2, 1, 1, 9, False, 1],
            [1, 144, 144, 59, 59, 5, 5, 2, 2, 0, 0, 144, False, 1],
            [1, 144, 144, 60, 60, 3, 3, 1, 1, 1, 1, 144, False, 1],
            [1, 144, 144, 63, 63, 5, 5, 2, 2, 0, 0, 144, False, 1],
            [1, 144, 144, 65, 65, 3, 3, 1, 1, 1, 1, 144, False, 1],
            [1, 144, 144, 69, 69, 5, 5, 2, 2, 0, 0, 144, False, 1],
            [1, 1512, 1512, 14, 14, 3, 3, 2, 2, 1, 1, 63, False, 1],
            [1, 1536, 1536, 10, 10, 3, 3, 1, 1, 1, 1, 1536, False, 1],
            [1, 8, 16, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 16, 16, 112, 112, 3, 3, 1, 1, 1, 1, 16, False, 1],
            [1, 16, 16, 112, 112, 3, 3, 2, 2, 1, 1, 16, False, 1],
            [1, 16, 16, 112, 112, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 8, 16, 112, 112, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 16, 16, 160, 160, 3, 3, 1, 1, 1, 1, 16, False, 1],
            [1, 16, 16, 160, 160, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 16, 16, 224, 224, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 32, 16, 224, 224, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 16, 16, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 160, 160, 14, 14, 3, 3, 1, 1, 1, 1, 10, False, 1],
            [1, 160, 160, 28, 28, 3, 3, 1, 1, 1, 1, 160, False, 1],
            [1, 160, 160, 28, 28, 3, 3, 2, 2, 1, 1, 10, False, 1],
            [1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, True, 1],
            [1, 1632, 1632, 12, 12, 3, 3, 1, 1, 1, 1, 1632, False, 1],
            [1, 1632, 1632, 12, 12, 5, 5, 1, 1, 2, 2, 1632, False, 1],
            [1, 168, 168, 28, 28, 3, 3, 1, 1, 1, 1, 7, False, 1],
            [1, 168, 168, 56, 56, 3, 3, 2, 2, 1, 1, 7, False, 1],
            [1, 184, 184, 14, 14, 3, 3, 1, 1, 1, 1, 184, False, 1],
            [1, 184, 184, 20, 20, 3, 3, 1, 1, 1, 1, 184, False, 1],
            [1, 184, 184, 7, 7, 1, 5, 1, 1, 0, 2, 184, False, 1],
            [1, 184, 184, 7, 7, 5, 1, 1, 1, 2, 0, 184, False, 1],
            [1, 192, 192, 14, 14, 3, 3, 1, 1, 1, 1, 192, False, 1],
            [1, 192, 192, 28, 28, 3, 3, 1, 1, 1, 1, 192, False, 1],
            [1, 192, 192, 28, 28, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 192, 192, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 48, 192, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 192, 192, 75, 75, 3, 3, 1, 1, 1, 1, 192, False, 1],
            [1, 192, 192, 79, 79, 5, 5, 2, 2, 0, 0, 192, False, 1],
            [1, 192, 192, 95, 95, 3, 3, 1, 1, 1, 1, 192, False, 1],
            [1, 192, 192, 99, 99, 5, 5, 2, 2, 0, 0, 192, False, 1],
            [1, 1920, 1920, 14, 14, 3, 3, 2, 2, 1, 1, 16, False, 1],
            [1, 20, 20, 28, 28, 3, 3, 1, 1, 1, 1, 20, False, 1],
            [1, 200, 200, 14, 14, 3, 3, 1, 1, 1, 1, 200, False, 1],
            [1, 200, 200, 20, 20, 3, 3, 1, 1, 1, 1, 200, False, 1],
            [1, 200, 200, 7, 7, 1, 5, 1, 1, 0, 2, 200, False, 1],
            [1, 200, 200, 7, 7, 5, 1, 1, 1, 2, 0, 200, False, 1],
            [1, 2016, 2016, 14, 14, 3, 3, 2, 2, 1, 1, 36, False, 1],
            [1, 2048, 2048, 14, 14, 3, 3, 2, 2, 1, 1, 16, False, 1],
            [1, 2048, 2048, 15, 20, 3, 3, 1, 1, 1, 1, 2048, True, 1],
            [1, 256, 2048, 23, 40, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 256, 2048, 25, 34, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 256, 2048, 25, 34, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 208, 208, 14, 14, 3, 3, 1, 1, 1, 1, 26, False, 1],
            [1, 208, 208, 28, 28, 3, 3, 2, 2, 1, 1, 26, False, 1],
            [1, 216, 216, 28, 28, 3, 3, 1, 1, 1, 1, 9, False, 1],
            [1, 216, 216, 56, 56, 3, 3, 2, 2, 1, 1, 9, False, 1],
            [1, 8, 224, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 224, 224, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 224, 224, 112, 112, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 224, 224, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 224, 224, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 224, 224, 56, 56, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 224, 224, 7, 7, 3, 3, 1, 1, 1, 1, 224, False, 1],
            [1, 8, 232, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 232, 232, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 232, 232, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 24, 24, 112, 112, 3, 3, 1, 1, 1, 1, 24, False, 1],
            [1, 24, 24, 56, 56, 5, 5, 2, 2, 2, 2, 24, False, 1],
            [1, 240, 240, 14, 14, 1, 5, 1, 1, 0, 2, 240, False, 1],
            [1, 240, 240, 14, 14, 3, 3, 1, 1, 1, 1, 240, False, 1],
            [1, 240, 240, 14, 14, 5, 1, 1, 1, 2, 0, 240, False, 1],
            [1, 240, 240, 14, 14, 5, 5, 1, 1, 2, 2, 240, False, 1],
            [1, 240, 240, 28, 28, 3, 3, 2, 2, 1, 1, 240, False, 1],
            [1, 240, 240, 28, 28, 5, 5, 1, 1, 2, 2, 240, False, 1],
            [1, 240, 240, 28, 28, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 240, 240, 29, 29, 3, 3, 2, 2, 0, 0, 240, False, 1],
            [1, 240, 240, 30, 30, 5, 5, 1, 1, 2, 2, 240, False, 1],
            [1, 240, 240, 31, 31, 3, 3, 2, 2, 0, 0, 240, False, 1],
            [1, 240, 240, 40, 40, 3, 3, 2, 2, 1, 1, 240, False, 1],
            [1, 240, 240, 56, 56, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 2520, 2520, 14, 14, 3, 3, 2, 2, 1, 1, 15, False, 1],
            [1, 256, 256, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 256, 256, 10, 10, 3, 3, 2, 2, 1, 1, 256, False, 1],
            [1, 256, 256, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, 256, True, 1],
            [1, 150, 256, 128, 128, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 256, 256, 13, 17, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 512, 256, 180, 320, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 512, 256, 19, 19, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 512, 256, 200, 272, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 256, 256, 25, 34, 3, 3, 1, 1, 1, 1, 1, True, 1],
            [1, 256, 256, 28, 28, 3, 3, 1, 1, 1, 1, 256, False, 1],
            [1, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 256, 256, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 256, 256, 56, 56, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 64, False, 1],
            [1, 256, 256, 56, 56, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 256, 256, 56, 56, 3, 3, 2, 2, 1, 1, 32, False, 1],
            [1, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 256, True, 1],
            [1, 256, 256, 75, 75, 3, 3, 1, 1, 1, 1, 256, False, 1],
            [1, 256, 256, 75, 75, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 288, 288, 14, 14, 5, 5, 2, 2, 2, 2, 288, False, 1],
            [1, 288, 288, 14, 14, 3, 3, 1, 1, 1, 1, 18, False, 1],
            [1, 288, 288, 28, 28, 3, 3, 2, 2, 1, 1, 18, False, 1],
            [1, 288, 288, 33, 33, 5, 5, 1, 1, 2, 2, 288, False, 1],
            [1, 288, 288, 35, 35, 3, 3, 2, 2, 0, 0, 288, False, 1],
            [1, 288, 288, 38, 38, 5, 5, 1, 1, 2, 2, 288, False, 1],
            [1, 288, 288, 39, 39, 3, 3, 2, 2, 0, 0, 288, False, 1],
            [1, 2904, 2904, 24, 24, 3, 3, 1, 1, 1, 1, 11, False, 1],
            [1, 2904, 2904, 48, 48, 3, 3, 2, 2, 1, 1, 11, False, 1],
            [1, 1024, 3, 224, 224, 16, 16, 16, 16, 0, 0, 1, True, 1],
            [1, 1024, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, True, 1],
            [1, 128, 3, 224, 224, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 16, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 16, 3, 224, 224, 7, 7, 1, 1, 3, 3, 1, False, 1],
            [1, 32, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 64, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 64, 3, 224, 224, 3, 3, 1, 1, 1, 1, 1, True, 1],
            [1, 64, 3, 224, 224, 7, 7, 2, 2, 3, 3, 1, False, 1],
            [1, 768, 3, 224, 224, 16, 16, 16, 16, 0, 0, 1, True, 1],
            [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, False, 1],
            [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, True, 1],
            [1, 96, 3, 224, 224, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 96, 3, 224, 224, 7, 7, 2, 2, 3, 3, 1, False, 1],
            [1, 32, 3, 225, 225, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 32, 3, 241, 241, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 128, 3, 256, 256, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 32, 3, 256, 256, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 96, 3, 256, 256, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 32, 3, 261, 261, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 32, 3, 299, 299, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 32, 3, 299, 299, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 64, 3, 300, 300, 3, 3, 1, 1, 1, 1, 1, True, 1],
            [1, 32, 3, 301, 301, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 16, 3, 320, 320, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 32, 3, 381, 381, 3, 3, 2, 2, 0, 0, 1, False, 1],
            [1, 32, 3, 384, 384, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 768, 3, 384, 512, 32, 32, 32, 32, 0, 0, 1, True, 1],
            [1, 64, 3, 480, 640, 7, 7, 4, 4, 3, 3, 1, True, 1],
            [1, 32, 3, 512, 512, 7, 7, 4, 4, 3, 3, 1, True, 1],
            [1, 192, 3, 512, 672, 16, 16, 16, 16, 0, 0, 1, True, 1],
            [1, 1280, 3, 518, 518, 14, 14, 14, 14, 0, 0, 1, True, 1],
            [1, 64, 3, 720, 1280, 7, 7, 2, 2, 3, 3, 1, False, 1],
            [1, 64, 3, 800, 1088, 7, 7, 2, 2, 3, 3, 1, False, 1],
            [1, 3024, 3024, 14, 14, 3, 3, 2, 2, 1, 1, 27, False, 1],
            [1, 16, 32, 112, 112, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 224, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 232, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 256, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 32, 32, 112, 112, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 32, 32, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 32, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 336, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 48, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 64, 32, 112, 112, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 72, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 80, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 96, 32, 112, 112, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 16, 32, 120, 120, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 32, 32, 120, 120, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, True, 1],
            [1, 64, 32, 128, 128, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 16, 32, 130, 130, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 32, 32, 130, 130, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 64, 32, 147, 147, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 32, 32, 149, 149, 3, 3, 1, 1, 0, 0, 1, False, 1],
            [1, 24, 32, 150, 150, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 32, 32, 150, 150, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 64, 32, 150, 150, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 24, 32, 190, 190, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 32, 32, 190, 190, 3, 3, 1, 1, 1, 1, 32, False, 1],
            [1, 528, 32, 192, 192, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 1, 32, 256, 256, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 32, 32, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 32, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 320, 320, 14, 14, 3, 3, 1, 1, 1, 1, 20, False, 1],
            [1, 320, 320, 28, 28, 3, 3, 2, 2, 1, 1, 20, False, 1],
            [1, 320, 320, 30, 40, 2, 2, 2, 2, 0, 0, 1, True, 1],
            [1, 336, 336, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 336, 336, 14, 14, 3, 3, 1, 1, 1, 1, 336, False, 1],
            [1, 336, 336, 14, 14, 3, 3, 1, 1, 1, 1, 14, False, 1],
            [1, 336, 336, 28, 28, 3, 3, 2, 2, 1, 1, 14, False, 1],
            [1, 336, 336, 48, 48, 5, 5, 1, 1, 2, 2, 336, False, 1],
            [1, 336, 336, 49, 49, 3, 3, 2, 2, 0, 0, 336, False, 1],
            [1, 336, 336, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 336, 336, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 36, 36, 56, 56, 3, 3, 1, 1, 1, 1, 36, False, 1],
            [1, 3712, 3712, 14, 14, 3, 3, 2, 2, 1, 1, 16, False, 1],
            [1, 384, 384, 14, 14, 3, 3, 1, 1, 1, 1, 384, False, 1],
            [1, 256, 384, 8, 8, 1, 3, 1, 1, 0, 1, 1, False, 1],
            [1, 256, 384, 8, 8, 3, 1, 1, 1, 1, 0, 1, False, 1],
            [1, 40, 40, 14, 14, 3, 3, 1, 1, 1, 1, 40, False, 1],
            [1, 40, 40, 28, 28, 3, 3, 2, 2, 1, 1, 40, False, 1],
            [1, 400, 400, 14, 14, 3, 3, 2, 2, 1, 1, 25, False, 1],
            [1, 400, 400, 7, 7, 3, 3, 1, 1, 1, 1, 25, False, 1],
            [1, 408, 408, 14, 14, 3, 3, 1, 1, 1, 1, 17, False, 1],
            [1, 408, 408, 28, 28, 3, 3, 2, 2, 1, 1, 17, False, 1],
            [1, 432, 432, 14, 14, 3, 3, 1, 1, 1, 1, 9, False, 1],
            [1, 432, 432, 28, 28, 3, 3, 2, 2, 1, 1, 9, False, 1],
            [1, 440, 440, 14, 14, 3, 3, 2, 2, 1, 1, 55, False, 1],
            [1, 440, 440, 7, 7, 3, 3, 1, 1, 1, 1, 55, False, 1],
            [1, 448, 448, 28, 28, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 448, 448, 28, 28, 3, 3, 1, 1, 1, 1, 8, False, 1],
            [1, 448, 448, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 448, 448, 56, 56, 3, 3, 2, 2, 1, 1, 8, False, 1],
            [1, 8, 48, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 48, 48, 112, 112, 3, 3, 2, 2, 1, 1, 48, False, 1],
            [1, 48, 48, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 48, 48, 112, 112, 3, 3, 2, 2, 1, 1, 6, False, 1],
            [1, 48, 48, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 48, 48, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 480, 480, 10, 10, 5, 5, 1, 1, 2, 2, 480, False, 1],
            [1, 480, 480, 14, 14, 3, 3, 1, 1, 1, 1, 480, False, 1],
            [1, 480, 480, 14, 14, 5, 5, 1, 1, 2, 2, 480, False, 1],
            [1, 480, 480, 15, 15, 3, 3, 1, 1, 1, 1, 480, False, 1],
            [1, 480, 480, 15, 15, 5, 5, 1, 1, 2, 2, 480, False, 1],
            [1, 480, 480, 20, 20, 3, 3, 1, 1, 1, 1, 480, False, 1],
            [1, 512, 512, 14, 14, 3, 3, 1, 1, 1, 1, 512, False, 1],
            [1, 64, 512, 15, 20, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 512, 512, 16, 16, 2, 2, 2, 2, 0, 0, 1, True, 1],
            [1, 1024, 512, 19, 19, 3, 3, 1, 1, 6, 6, 1, True, 6],
            [1, 512, 512, 28, 28, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 512, 512, 5, 5, 3, 3, 1, 1, 1, 1, 512, False, 1],
            [1, 512, 512, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 512, 512, 56, 56, 3, 3, 2, 2, 1, 1, 32, False, 1],
            [1, 512, 512, 56, 56, 3, 3, 2, 2, 1, 1, 64, False, 1],
            [1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, 512, True, 1],
            [1, 8, 528, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 528, 528, 17, 17, 3, 3, 1, 1, 1, 1, 528, False, 1],
            [1, 528, 528, 17, 17, 5, 5, 1, 1, 2, 2, 528, False, 1],
            [1, 528, 528, 192, 192, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 528, 528, 96, 96, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 528, 528, 96, 96, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 56, 56, 14, 14, 3, 3, 1, 1, 1, 1, 56, False, 1],
            [1, 576, 576, 14, 14, 3, 3, 1, 1, 1, 1, 576, False, 1],
            [1, 576, 576, 14, 14, 3, 3, 1, 1, 1, 1, 24, False, 1],
            [1, 576, 576, 19, 19, 3, 3, 1, 1, 1, 1, 576, False, 1],
            [1, 576, 576, 19, 19, 5, 5, 1, 1, 2, 2, 576, False, 1],
            [1, 576, 576, 28, 28, 3, 3, 2, 2, 1, 1, 24, False, 1],
            [1, 576, 576, 7, 7, 5, 5, 1, 1, 2, 2, 576, False, 1],
            [1, 60, 60, 28, 28, 3, 3, 1, 1, 1, 1, 60, False, 1],
            [1, 8, 64, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 64, 64, 112, 112, 3, 3, 1, 1, 1, 1, 64, False, 1],
            [1, 64, 64, 112, 112, 3, 3, 2, 2, 1, 1, 64, False, 1],
            [1, 64, 64, 112, 112, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 64, 64, 112, 112, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 128, 64, 120, 160, 3, 3, 2, 2, 1, 1, 1, True, 1],
            [1, 64, 64, 120, 160, 8, 8, 8, 8, 0, 0, 1, True, 1],
            [1, 128, 64, 150, 150, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 64, 150, 150, 3, 3, 1, 1, 1, 1, 64, False, 1],
            [1, 64, 64, 160, 160, 3, 3, 2, 2, 1, 1, 64, False, 1],
            [1, 64, 64, 180, 320, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 64, 180, 320, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 64, 64, 200, 272, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 64, 200, 272, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 128, 64, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 128, 64, 56, 56, 1, 1, 2, 2, 0, 0, 1, False, 1],
            [1, 128, 64, 56, 56, 3, 3, 2, 2, 1, 1, 1, False, 1],
            [1, 192, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 256, 64, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 64, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 64, 64, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, True, 1],
            [1, 64, 64, 73, 73, 1, 7, 1, 1, 0, 3, 1, False, 1],
            [1, 64, 64, 73, 73, 7, 1, 1, 1, 3, 0, 1, False, 1],
            [1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, True, 1],
            [1, 672, 672, 14, 14, 3, 3, 1, 1, 1, 1, 672, False, 1],
            [1, 672, 672, 14, 14, 5, 5, 1, 1, 2, 2, 672, False, 1],
            [1, 672, 672, 14, 14, 5, 5, 2, 2, 2, 2, 672, False, 1],
            [1, 672, 672, 14, 14, 3, 3, 2, 2, 1, 1, 42, False, 1],
            [1, 672, 672, 15, 15, 5, 5, 1, 1, 2, 2, 672, False, 1],
            [1, 672, 672, 17, 17, 5, 5, 2, 2, 0, 0, 672, False, 1],
            [1, 672, 672, 19, 19, 5, 5, 2, 2, 0, 0, 672, False, 1],
            [1, 672, 672, 20, 20, 3, 3, 1, 1, 1, 1, 672, False, 1],
            [1, 672, 672, 20, 20, 5, 5, 2, 2, 2, 2, 672, False, 1],
            [1, 672, 672, 24, 24, 3, 3, 1, 1, 1, 1, 672, False, 1],
            [1, 672, 672, 24, 24, 5, 5, 1, 1, 2, 2, 672, False, 1],
            [1, 672, 672, 28, 28, 3, 3, 1, 1, 1, 1, 4, False, 1],
            [1, 672, 672, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],
            [1, 672, 672, 7, 7, 1, 5, 1, 1, 0, 2, 672, False, 1],
            [1, 672, 672, 7, 7, 5, 1, 1, 1, 2, 0, 672, False, 1],
            [1, 672, 672, 7, 7, 3, 3, 1, 1, 1, 1, 42, False, 1],
            [1, 696, 696, 28, 28, 3, 3, 1, 1, 1, 1, 3, False, 1],
            [1, 696, 696, 56, 56, 3, 3, 2, 2, 1, 1, 3, False, 1],
            [1, 20, 72, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 24, 72, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 8, 72, 1, 1, 1, 1, 1, 1, 0, 0, 1, True, 1],
            [1, 72, 72, 112, 112, 3, 3, 2, 2, 1, 1, 3, False, 1],
            [1, 72, 72, 28, 28, 1, 5, 1, 1, 0, 2, 72, False, 1],
            [1, 72, 72, 28, 28, 5, 1, 1, 1, 2, 0, 72, False, 1],
            [1, 72, 72, 56, 56, 3, 3, 1, 1, 1, 1, 72, False, 1],
            [1, 72, 72, 56, 56, 3, 3, 2, 2, 1, 1, 72, False, 1],
            [1, 72, 72, 56, 56, 5, 5, 2, 2, 2, 2, 72, False, 1],
            [1, 72, 72, 56, 56, 3, 3, 1, 1, 1, 1, 3, False, 1],
            [1, 72, 72, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 72, 72, 80, 80, 3, 3, 1, 1, 1, 1, 72, False, 1],
            [1, 72, 72, 80, 80, 5, 5, 2, 2, 2, 2, 72, False, 1],
            [1, 720, 720, 14, 14, 3, 3, 1, 1, 1, 1, 6, False, 1],
            [1, 720, 720, 17, 17, 5, 5, 1, 1, 2, 2, 720, False, 1],
            [1, 720, 720, 21, 21, 5, 5, 2, 2, 0, 0, 720, False, 1],
            [1, 720, 720, 28, 28, 3, 3, 2, 2, 1, 1, 6, False, 1],
            [1, 728, 728, 38, 38, 3, 3, 1, 1, 1, 1, 728, False, 1],
            [1, 7392, 7392, 24, 24, 3, 3, 2, 2, 1, 1, 28, False, 1],
            [1, 784, 784, 14, 14, 3, 3, 2, 2, 1, 1, 49, False, 1],
            [1, 784, 784, 7, 7, 3, 3, 1, 1, 1, 1, 49, False, 1],
            [1, 8, 8, 112, 112, 3, 3, 1, 1, 1, 1, 8, False, 1],
            [1, 80, 80, 14, 14, 3, 3, 1, 1, 1, 1, 80, False, 1],
            [1, 80, 80, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 80, 80, 56, 56, 3, 3, 1, 1, 1, 1, 1, False, 1],
            [1, 816, 816, 19, 19, 5, 5, 1, 1, 2, 2, 816, False, 1],
            [1, 816, 816, 23, 23, 5, 5, 2, 2, 0, 0, 816, False, 1],
            [1, 88, 88, 28, 28, 3, 3, 1, 1, 1, 1, 88, False, 1],
            [1, 888, 888, 14, 14, 3, 3, 2, 2, 1, 1, 37, False, 1],
            [1, 888, 888, 7, 7, 3, 3, 1, 1, 1, 1, 37, False, 1],
            [1, 896, 896, 14, 14, 3, 3, 1, 1, 1, 1, 7, False, 1],
            [1, 896, 896, 14, 14, 3, 3, 1, 1, 1, 1, 16, False, 1],
            [1, 896, 896, 28, 28, 3, 3, 2, 2, 1, 1, 7, False, 1],
            [1, 896, 896, 28, 28, 3, 3, 2, 2, 1, 1, 16, False, 1],
            [1, 912, 912, 14, 14, 3, 3, 2, 2, 1, 1, 38, False, 1],
            [1, 912, 912, 7, 7, 3, 3, 1, 1, 1, 1, 38, False, 1],
            [1, 92, 92, 14, 14, 3, 3, 1, 1, 1, 1, 92, False, 1],
            [1, 96, 96, 112, 112, 3, 3, 2, 2, 1, 1, 96, False, 1],
            [1, 96, 96, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],
            [1, 96, 96, 113, 113, 3, 3, 2, 2, 0, 0, 96, False, 1],
            [1, 96, 96, 121, 121, 3, 3, 2, 2, 0, 0, 96, False, 1],
            [1, 96, 96, 131, 131, 3, 3, 2, 2, 0, 0, 96, False, 1],
            [1, 96, 96, 28, 28, 5, 5, 2, 2, 2, 2, 96, False, 1],
            [1, 192, 96, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 96, 96, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],
            [1, 96, 96, 56, 56, 1, 1, 1, 1, 0, 0, 1, False, 1],
            [1, 960, 960, 24, 24, 5, 5, 1, 1, 2, 2, 960, False, 1],
            [1, 960, 960, 27, 27, 5, 5, 2, 2, 0, 0, 960, False, 1],
            [1, 960, 960, 3, 3, 1, 5, 1, 1, 0, 2, 960, False, 1],
            [1, 960, 960, 3, 3, 5, 1, 1, 1, 2, 0, 960, False, 1],
            [1, 960, 960, 7, 7, 3, 3, 1, 1, 1, 1, 960, False, 1],
            [1, 960, 960, 7, 7, 5, 5, 1, 1, 2, 2, 960, False, 1],
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    *,
    device,
) -> list:
    return run_short(
        input_specs,
        device,
    )


import pytest


@pytest.mark.parametrize("input_spec", parameters["short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_localrun(device, input_spec):
    run_short(
        input_spec,
        device,
    )


failing_parameters = [
    # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_x, stride_y, pad_x, pad_y, groups, bias, dilation]
    # Input is 32MB maps to MM 64 cores, we neeed to avoid sharding this tensor and use dram intrelaved directly with MM
    [1, 256, 1024, 128, 128, 1, 1, 1, 1, 0, 0, 1, False, 1],  # 6
    [1, 1056, 1056, 48, 48, 3, 3, 1, 1, 1, 1, 4, False, 1],  # 14
    [1, 1056, 1056, 96, 96, 3, 3, 2, 2, 1, 1, 4, False, 1],  # 15
    [1, 192, 192, 99, 99, 5, 5, 2, 2, 0, 0, 192, False, 1],  # 100
    [1, 2520, 2520, 14, 14, 3, 3, 2, 2, 1, 1, 15, False, 1],  # 141
    [1, 2904, 2904, 24, 24, 3, 3, 1, 1, 1, 1, 11, False, 1],  # 170
    [1, 2904, 2904, 48, 48, 3, 3, 2, 2, 1, 1, 11, False, 1],  # 171
    [1, 1024, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, True, 1],  # 173
    [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, False, 1],  # 182
    [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, True, 1],  # 183
    [1, 768, 3, 384, 512, 32, 32, 32, 32, 0, 0, 1, True, 1],  # 199
    [1, 64, 3, 800, 1088, 7, 7, 2, 2, 3, 3, 1, False, 1],  # 205
    [1, 336, 336, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],  # 241
    [1, 336, 336, 48, 48, 5, 5, 1, 1, 2, 2, 336, False, 1],  # 245
    [1, 336, 336, 56, 56, 3, 3, 1, 1, 1, 1, 2, False, 1],  # 247
    [1, 528, 528, 17, 17, 5, 5, 1, 1, 2, 2, 528, False, 1],  # 292
    [1, 528, 528, 192, 192, 3, 3, 2, 2, 1, 1, 2, False, 1],  # 293
    [1, 528, 528, 96, 96, 3, 3, 1, 1, 1, 1, 2, False, 1],  # 294
    [1, 576, 576, 19, 19, 5, 5, 1, 1, 2, 2, 576, False, 1],  # 300
    [1, 672, 672, 24, 24, 5, 5, 1, 1, 2, 2, 672, False, 1],  # 341
    [1, 696, 696, 28, 28, 3, 3, 1, 1, 1, 1, 3, False, 1],  # 347
    [1, 696, 696, 56, 56, 3, 3, 2, 2, 1, 1, 3, False, 1],  # 348
    [1, 720, 720, 17, 17, 5, 5, 1, 1, 2, 2, 720, False, 1],  # 363
    [1, 728, 728, 38, 38, 3, 3, 1, 1, 1, 1, 728, False, 1],  # 366
    [1, 7392, 7392, 24, 24, 3, 3, 2, 2, 1, 1, 28, False, 1],  # 367
    [1, 816, 816, 19, 19, 5, 5, 1, 1, 2, 2, 816, False, 1],  # 374
    [1, 960, 960, 24, 24, 5, 5, 1, 1, 2, 2, 960, False, 1],  # 395
]


@pytest.mark.parametrize("input_spec", failing_parameters)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_localrun_fail_only(device, input_spec):
    run_short(
        input_spec,
        device,
    )
