# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [8, "TTNN-BERT_LARGE-batch_8-MIXED_PRECISION_BATCH8", 165],
    ],
)
def test_perf_device_bare_metal(batch_size, test, expected_perf):
    subdir = "ttnn_bert"
    num_iterations = 4
    margin = 0.03
    command = f"pytest models/demos/bert/demo/demo.py::test_demo_squadv2[3-models.demos.bert.tt.ttnn_optimized_bert-phiyodr/bert-large-finetuned-squad2]"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_bert11",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )
