# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import time
from models.utility_functions import pad_and_fold_conv_activation_for_unity_stride, pad_and_fold_act_2


def test_pad_fold():
    torch_input_tensor = torch.randint(-10, 10, (8, 3, 224, 224), dtype=torch.float32)

    start_time = time.time()
    input_tensor = pad_and_fold_conv_activation_for_unity_stride(torch_input_tensor, 3, 3, 2, 2)
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    print("Old time taken is ", time.time() - start_time, "s ")
    start_time = time.time()
    input_tensor2 = pad_and_fold_act_2(torch_input_tensor, 3, 3, 2, 2)
    print("New time taken is ", time.time() - start_time, "s ")
    print(input_tensor.shape)
    print(input_tensor2.shape)

    assert torch.equal(input_tensor, input_tensor2)
