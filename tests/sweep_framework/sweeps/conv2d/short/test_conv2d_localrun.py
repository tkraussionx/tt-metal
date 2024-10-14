import pytest
from tests.sweep_framework.sweeps.conv2d.short.conv2d_short_sweep import parameters
from tests.sweep_framework.sweep_utils.conv2d_common import run_short


@pytest.mark.parametrize("input_spec", parameters["short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_localrun(device, input_spec):
    run_short(
        input_spec,
        device,
    )


parameters_fail = [
    # Contains following params
    # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_x, stride_y, pad_x, pad_y, groups, bias, dilation]
    [
        1,
        256,
        1024,
        128,
        128,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        False,
        1,
    ],  # HS 8x8 grid out of memory l1 CB, run MM w/o sharding the input
    [1, 1024, 1024, 19, 19, 1, 1, 1, 1, 0, 0, 1, True, 1],  # Out of memory run MM w/o sharding the input
    [1, 2048, 1024, 7, 7, 1, 1, 1, 1, 0, 0, 1, True, 1],  # PASS with act_block_w_ntiles == 1
    [1, 1056, 1056, 48, 48, 3, 3, 1, 1, 1, 1, 4, False, 1],  # OOM 21 cores BS 1056 -> 33 tiles
    [1, 1056, 1056, 96, 96, 3, 3, 2, 2, 1, 1, 4, False, 1],  # OOM 24 cores
    [1, 1152, 1152, 7, 7, 5, 5, 1, 1, 2, 2, 1152, False, 1],  # OOM width shard forced to HS due to 5x5 kernel
    [
        1,
        1248,
        1248,
        9,
        9,
        5,
        5,
        1,
        1,
        2,
        2,
        1248,
        False,
        1,
    ],  # nc risk compile fail, width shard forced to HS due to 5x5 kernel
    [1, 128, 128, 180, 320, 3, 3, 2, 2, 1, 1, 1, False, 1],  # PASS with act_block_h_override=32
    [1, 128, 128, 200, 272, 3, 3, 2, 2, 1, 1, 1, False, 1],  # OOM HHW = 425 tiles which maps to 25 cores
    [1, 1280, 1280, 30, 40, 3, 3, 1, 1, 1, 1, 1280, True, 1],  # PASS with act_block_h_override=32
    [
        1,
        1392,
        1392,
        10,
        10,
        5,
        5,
        1,
        1,
        2,
        2,
        1392,
        False,
        1,
    ],  # OOM width shard forced to HS due to 5x5 kernels, maps to 4 cores
    [1, 144, 144, 191, 191, 3, 3, 2, 2, 0, 0, 144, False, 1],  # OOM NHW = 283 tiles (prime number) maps to single core
    [1, 144, 144, 60, 60, 3, 3, 1, 1, 1, 1, 144, False, 1],  # OOM NHW = 113 tiles (prime number) maps to single core
    [
        1,
        1632,
        1632,
        12,
        12,
        5,
        5,
        1,
        1,
        2,
        2,
        1632,
        False,
        1,
    ],  # OOM width shard forced to HS due to 5x5 kernels, maps to 5 cores
    [1, 192, 192, 95, 95, 3, 3, 1, 1, 1, 1, 192, False, 1],  # OOM NHW = 283 tiles (prime number) maps to single core
    [
        1,
        256,
        2048,
        25,
        34,
        3,
        3,
        2,
        2,
        1,
        1,
        1,
        True,
        1,
    ],  # OOM Auto shard heuristic pics bad WS, passes with BS 8 vs 56 cores
    [1, 2520, 2520, 14, 14, 3, 3, 2, 2, 1, 1, 15, False, 1],  # OOM 1 core 2520 -> 79 tiles (prime number) -> 1 core
    [
        1,
        256,
        256,
        120,
        160,
        3,
        3,
        1,
        1,
        1,
        1,
        256,
        True,
        1,
    ],  # OOM auto_shards picks HS and weights are too big to fit, BS sharding solves the issue.
    [
        1,
        512,
        256,
        180,
        320,
        1,
        1,
        2,
        2,
        0,
        0,
        1,
        False,
        1,
    ],  # OOM auto_shards picks HS and weights are too big to fit, BS sharding solves the issue.
    [
        1,
        512,
        256,
        200,
        272,
        1,
        1,
        2,
        2,
        0,
        0,
        1,
        False,
        1,
    ],  # OOM auto_shards picks HS and weights are too big to fit, BS sharding solves the issue.
    [
        1,
        2904,
        2904,
        24,
        24,
        3,
        3,
        1,
        1,
        1,
        1,
        11,
        False,
        1,
    ],  # OOM auto_shards picks HS, BS sharding solves the issue 25 vs 64 cores.
    [1, 2904, 2904, 48, 48, 3, 3, 2, 2, 1, 1, 11, False, 1],  # OOM BS L1 out of memmory
    [
        1,
        1024,
        3,
        224,
        224,
        16,
        16,
        16,
        16,
        0,
        0,
        1,
        True,
        1,
    ],  # OOM WS forced to HS due to 16x16 kernel 7 cores (max in/out channels)
    [
        1,
        1024,
        3,
        224,
        224,
        32,
        32,
        32,
        32,
        0,
        0,
        1,
        True,
        1,
    ],  # OOM WS forced to HS due to 32x32 kernel 2 cores (max in/out channels)
    [1, 768, 3, 224, 224, 16, 16, 16, 16, 0, 0, 1, True, 1],  # OOM WS forced to HS due to 16x16
    [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, False, 1],  # OOM WS forced to HS due to 32x32
    [1, 768, 3, 224, 224, 32, 32, 32, 32, 0, 0, 1, True, 1],  # OOM WS forced to HS due to 32x32
    [1, 32, 3, 299, 299, 3, 3, 2, 2, 0, 0, 1, False, 1],  # OOM NHW = 694, 2 cores as 347 is a prime number
    [1, 64, 3, 300, 300, 3, 3, 1, 1, 1, 1, 1, True, 1],  # Pass with act_block_h_override=32
    [1, 32, 3, 381, 381, 3, 3, 2, 2, 0, 0, 1, False, 1],  # OOM NHW = 1129 (prime), 1 core
    [1, 768, 3, 384, 512, 32, 32, 32, 32, 0, 0, 1, True, 1],  # OOM WS forced to HS due to 32x32
    [1, 192, 3, 512, 672, 16, 16, 16, 16, 0, 0, 1, True, 1],  # OOM BS forced to HS due to 16x16
    [1, 1280, 3, 518, 518, 14, 14, 14, 14, 0, 0, 1, True, 1],  # OOM BS forced to HS due to 14x14
    [1, 64, 3, 720, 1280, 7, 7, 2, 2, 3, 3, 1, False, 1],  # OOM Halo allocation fail large image and kernel.
    [1, 64, 3, 800, 1088, 7, 7, 2, 2, 3, 3, 1, False, 1],  # OOM Halo allocation fail large image and kernel.
    [1, 24, 32, 190, 190, 1, 1, 1, 1, 0, 0, 1, False, 1],  # OOM NHW = 1129 (prime), 1 core
    [1, 32, 32, 190, 190, 3, 3, 1, 1, 1, 1, 32, False, 1],  # OOM NHW = 1129 (prime), 1 core
    [1, 336, 336, 112, 112, 3, 3, 2, 2, 1, 1, 2, False, 1],  # OOM HS 49 cores
    [1, 336, 336, 14, 14, 3, 3, 1, 1, 1, 1, 336, False, 1],  # PASS with removed assert
    [1, 336, 336, 14, 14, 3, 3, 1, 1, 1, 1, 14, False, 1],  # PASS with removed assert
    [1, 336, 336, 28, 28, 3, 3, 2, 2, 1, 1, 14, False, 1],  # PASS with removed assert
    [1, 336, 336, 48, 48, 5, 5, 1, 1, 2, 2, 336, False, 1],  # OOM BS forced to HS due to 5x5
    [1, 336, 336, 49, 49, 3, 3, 2, 2, 0, 0, 336, False, 1],  # PASS with removed assert
    [1, 480, 480, 10, 10, 5, 5, 1, 1, 2, 2, 480, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 480, 480, 14, 14, 5, 5, 1, 1, 2, 2, 480, False, 1],  # OOM BS forced to HS due to 5x5
    [1, 480, 480, 15, 15, 5, 5, 1, 1, 2, 2, 480, False, 1],  # OOM BS forced to HS due to 5x5
    [1, 512, 512, 16, 16, 2, 2, 2, 2, 0, 0, 1, True, 1],  # OOM WS forced to HS due to 2x2
    [
        1,
        512,
        512,
        60,
        80,
        3,
        3,
        1,
        1,
        1,
        1,
        512,
        True,
        1,
    ],  # OOM auto_hard selects HS and should select BS, with BS and act_block_h_override = 32 is PASSES
    [1, 528, 528, 17, 17, 3, 3, 1, 1, 1, 1, 528, False, 1],  # OOM BS Output 2D Matrix Width tiles: 17 (prime number)
    [
        1,
        528,
        528,
        17,
        17,
        5,
        5,
        1,
        1,
        2,
        2,
        528,
        False,
        1,
    ],  # OOM BS forced to HS due to 5x5, but even BS fails as Output 2D Matrix Width tiles: 17 (prime number)
    [
        1,
        528,
        528,
        192,
        192,
        3,
        3,
        2,
        2,
        1,
        1,
        2,
        False,
        1,
    ],  # OOM HS, BS should be better but Output 2D Matrix Width tiles: 17 (prime)
    [
        1,
        528,
        528,
        96,
        96,
        3,
        3,
        1,
        1,
        1,
        1,
        2,
        False,
        1,
    ],  # OOM HS, BS should be better but Output 2D Matrix Width tiles: 17 (prime)
    [1, 576, 576, 19, 19, 5, 5, 1, 1, 2, 2, 576, False, 1],  # OOM BS forced to HS due to 5x5
    [1, 576, 576, 7, 7, 5, 5, 1, 1, 2, 2, 576, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 64, 64, 73, 73, 1, 7, 1, 1, 0, 3, 1, False, 1],  # OOM HS NHW = 167 (prime)
    [1, 64, 64, 73, 73, 7, 1, 1, 1, 3, 0, 1, False, 1],  # OOM HS NHW = 167 (prime)
    [1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, True, 1],  # PASS with act_block_h_override=32
    [1, 672, 672, 14, 14, 5, 5, 1, 1, 2, 2, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 14, 14, 5, 5, 2, 2, 2, 2, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 15, 15, 5, 5, 1, 1, 2, 2, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 17, 17, 5, 5, 2, 2, 0, 0, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 19, 19, 5, 5, 2, 2, 0, 0, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 20, 20, 5, 5, 2, 2, 2, 2, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 24, 24, 5, 5, 1, 1, 2, 2, 672, False, 1],  # OOM WS forced to HS due to 5x5
    [1, 672, 672, 56, 56, 3, 3, 2, 2, 1, 1, 4, False, 1],  # PASS with act_block_h_override=32
    [1, 672, 672, 7, 7, 1, 5, 1, 1, 0, 2, 672, False, 1],  # OOM WS forced to HS due to 1x5
    [1, 696, 696, 28, 28, 3, 3, 1, 1, 1, 1, 3, False, 1],  # OOM BS num width tiles = 22  maps to just two row on width
    [1, 696, 696, 56, 56, 3, 3, 2, 2, 1, 1, 3, False, 1],  # OOM BS num width tiles = 22  maps to just two row on width
    [
        1,
        720,
        720,
        17,
        17,
        5,
        5,
        1,
        1,
        2,
        2,
        720,
        False,
        1,
    ],  # OOM BS forced to HS due to 5x5, but even BS would fail as num width tiles = 23 (prime number)
    [
        1,
        720,
        720,
        21,
        21,
        5,
        5,
        2,
        2,
        0,
        0,
        720,
        False,
        1,
    ],  # OOM WS forced to HS due to 5x5, but even BS would fail as num width tiles = 23 (prime number)
    [
        1,
        728,
        728,
        38,
        38,
        3,
        3,
        1,
        1,
        1,
        1,
        728,
        False,
        1,
    ],  # OOM BS num width tiles = 23  (prime number) HS would be better
    [1, 7392, 7392, 24, 24, 3, 3, 2, 2, 1, 1, 28, False, 1],  # OOM WS num width tiles = 231 (prime number)
    [1, 816, 816, 19, 19, 5, 5, 1, 1, 2, 2, 816, False, 1],  # nc risk compile fail, BS forced to HS due to 5x5 kernel
    [1, 816, 816, 23, 23, 5, 5, 2, 2, 0, 0, 816, False, 1],  # nc risk compile fail, WS forced to HS due to 5x5 kernel
    [1, 96, 96, 121, 121, 3, 3, 2, 2, 0, 0, 96, False, 1],  # OOM HS NHW = 113 (prime)
    [1, 960, 960, 24, 24, 5, 5, 1, 1, 2, 2, 960, False, 1],  # nc risk compile fail, BS forced to HS due to 5x5 kernel
    [1, 960, 960, 27, 27, 5, 5, 2, 2, 0, 0, 960, False, 1],  # nc risk compile fail, WS forced to HS due to 5x5 kernel
    [1, 960, 960, 3, 3, 1, 5, 1, 1, 0, 2, 960, False, 1],  # nc risk compile fail, WS forced to HS due to 1x5 kernel
    [1, 960, 960, 3, 3, 5, 1, 1, 1, 2, 0, 960, False, 1],  # OOM WS forced to HS due to 5x1, 1 core
    [1, 960, 960, 7, 7, 5, 5, 1, 1, 2, 2, 960, False, 1],  # nc risk compile fail, WS forced to HS due to 5x5 kernel
]


@pytest.mark.parametrize("input_spec", parameters_fail)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_localrun_fail_only(device, input_spec):
    run_short(
        input_spec,
        device,
    )
