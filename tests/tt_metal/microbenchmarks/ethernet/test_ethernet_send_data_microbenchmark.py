# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def report_results(test_case_name, num_messages_to_send, buffer_size_bytes):
    print("repoting results...")
    setup = device_post_proc_config.default_setup()
    setup.timerAnalysis = {
        "LATENCY": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "timerID": 12},
            "end": {"risc": "ERISC", "timerID": 3},
        },
    }

    throughput_GB_per_second = -1
    os.system("sed -i '/^[[:space:]]*$/d' ./generated/profiler/.logs/profile_log_device.csv")
    try:
        setup.deviceInputLog = "./generated/profiler/.logs/profile_log_device.csv"
        stats = import_log_run_stats(setup)
        core = [key for key in stats["devices"][1]["cores"].keys() if key != "DEVICE"][0]
        # test_cycles = stats["devices"][1]["cores"][core]["riscs"]["ERISC"]["analysis"]["LATENCY"]["stats"]["First"]
        test_cycles = stats["devices"][1]["cores"][core]["riscs"]["TENSIX"]["analysis"]["LATENCY"]["stats"]["First"]
        total_bytes = num_messages_to_send * buffer_size_bytes
        cycles_per_second = 1000000000
        throughput_bytes_per_second = total_bytes / (test_cycles / 1000000000)
        throughput_GB_per_second = throughput_bytes_per_second / (1000 * 1000 * 1000)
        print(
            f"Cycles: {test_cycles}, Buffer size(B): {buffer_size_bytes}, Loops: {num_messages_to_send}, Throughput: {throughput_GB_per_second} GB/s"
        )
    except:
        print("Error in results parsing")
        breakpoint()
        assert False

    os.system(f"rm -rf ./generated/profiler/.logs/{test_case_name}/")
    os.system(f"mkdir -p ./generated/profiler/.logs/{test_case_name}/")
    os.system(
        f"cp ./generated/profiler/.logs/profile_log_device.csv ./generated/profiler/.logs/{test_case_name}/profile_log_device.csv"
    )

    return throughput_GB_per_second


@pytest.mark.parametrize("num_messages_to_send", [1, 2, 4, 8, 16, 326, 680, 1000])
@pytest.mark.parametrize(
    "buffer_size_bytes",
    [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16 * 1024,
        24 * 1024,
        32 * 1024,
        50 * 1024,
        64 * 1024,
        128 * 1024,
    ],
)
@pytest.mark.parametrize("num_sends_before_sync", [1, 2, 4, 8])
def test_ethernet_send_data_microbenchmark(num_messages_to_send, buffer_size_bytes, num_sends_before_sync):
    print(
        f"test_ethernet_send_data_microbenchmark - buffer_size_bytes: {buffer_size_bytes}, num_messages_to_send: {num_messages_to_send}, num_sends_before_sync: {num_sends_before_sync}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_sends_before_sync}  \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping.cpp\" > /dev/null 2>&1"
    )
    if rc != 0:
        print("Error in running the test")
        assert False
    return report_results(
        f"test_ethernet_send_data_microbenchmark_{num_messages_to_send}_{buffer_size_bytes}_{num_sends_before_sync}",
        num_messages_to_send,
        buffer_size_bytes,
    )


@pytest.mark.parametrize("num_messages_to_send", [1, 2, 4, 8, 16, 100, 326, 680, 1000])
@pytest.mark.parametrize(
    "buffer_size_bytes",
    [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16 * 1024, 24 * 1024, 32 * 1024, 50 * 1024, 64 * 1024, 128 * 1024],
)
@pytest.mark.parametrize("num_transaction_buffers", [1, 2, 4, 8])
def test_ethernet_send_data_microbenchmark_concurrent(num_messages_to_send, buffer_size_bytes, num_transaction_buffers):
    print(
        f"test_ethernet_send_data_microbenchmark - buffer_size_bytes: {buffer_size_bytes}, num_messages_to_send: {num_messages_to_send}, num_transaction_buffers: {num_transaction_buffers}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        # f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\" > /dev/null 2>&1"
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\""
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    return report_results(
        f"test_ethernet_send_data_microbenchmark_concurrent_{num_messages_to_send}_{buffer_size_bytes}_{num_transaction_buffers}",
        num_messages_to_send,
        buffer_size_bytes,
    )


@pytest.mark.parametrize("num_messages_to_send", [2, 4, 8, 16, 100, 326, 680, 1000])
@pytest.mark.parametrize(
    "buffer_size_bytes",
    [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16 * 1024, 32 * 1024, 50 * 1024, 64 * 1024, 128 * 1024],
)
@pytest.mark.parametrize("num_transaction_buffers", [1, 2, 4, 8])
def test_ethernet_send_data_microbenchmark_concurrent_with_dram_read(
    num_messages_to_send, buffer_size_bytes, num_transaction_buffers
):
    print(
        f"test_ethernet_send_data_microbenchmark - buffer_size_bytes: {buffer_size_bytes}, num_messages_to_send: {num_messages_to_send}, num_transaction_buffers: {num_transaction_buffers}"
    )
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")
    rc = os.system(
        # f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_forward_local_chip_data_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\" > /dev/null 2>&1"
        f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_send_data_looping {buffer_size_bytes} {num_messages_to_send} {num_transaction_buffers} \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_forward_local_chip_data_looping_multi_channel.cpp\" \"tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_looping_multi_channel.cpp\""
    )
    if rc != 0:
        print("Error in running the test")
        assert False

    return report_results(
        f"test_ethernet_send_data_microbenchmark_concurrent_{num_messages_to_send}_{buffer_size_bytes}_{num_transaction_buffers}",
        num_messages_to_send,
        buffer_size_bytes,
    )


def run_ethernet_send_data_microbenchmark_sweep():
    loop_counts = [1, 2, 4, 8, 16, 100, 1000]
    buffer_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]
    num_parallel_buffers = [1, 2, 4, 8]

    recorded_throughput_slow_mode = {}
    recorded_throughput_concurrent = {}

    for num_messages_to_send in loop_counts:
        for buffer_size in buffer_sizes:
            for num_buffers in num_parallel_buffers:
                throughput_looping_synced = test_ethernet_send_data_microbenchmark(num_messages_to_send, buffer_size)
                throughput_looping_concurrent = test_ethernet_send_data_microbenchmark_concurrent(
                    num_messages_to_send, buffer_size, num_parallel_buffer
                )

                recorded_throughput_slow_mode[num_messages_to_send][buffer_size][
                    num_buffers
                ] = throughput_looping_synced
                recorded_throughput_concurrent[num_messages_to_send][buffer_size][
                    num_buffers
                ] = throughput_looping_concurrent

    for num_messages_to_send in loop_counts:
        for buffer_size in buffer_sizes:
            for num_buffers in num_parallel_buffers:
                pass
