
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

for ((i=1; i<=1000; i++))
do
    echo "============ Running command for iteration $i ============"
    if ! timeout 400s pytest -svv tests/ttnn/unit_tests/operations/test_conv2d.py::test_sd_conv_wh[enable_auto_formatting=True-math_fidelity=MathFidelity.LoFi-fp32_accum=False-activations_dtype=DataType.BFLOAT16-weights_dtype=DataType.BFLOAT8_B-batch_size=2-output_channels=320-input_channels=16-input_height=64-input_width=64-filter_height=3-filter_width=3-stride_h=1-stride_w=1-pad_h=1-pad_w=1-use_1d_systolic_array=True-config_override=None-device_l1_small_size=16384]; then
        echo "Command hung at iteration $i"
        break
    fi
done
