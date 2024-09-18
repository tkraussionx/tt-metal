# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

device = ttnn.open_device(0)
input = ttnn.ones([1, 1, 32, 32]).to(device)
input = ttnn.tilize_with_zero_padding(input)
input = ttnn.multiply(input, -3.1415)
output = ttnn.sin(input)
ttnn.set_printoptions(profile="full")
print(input)
print(output)
print("=========================== standard output =======================")
print(math.sin(-3.14062))
