# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn
import matplotlib.pyplot as plt

import math
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

TILE_WIDTH = 32
TILE_HEIGHT = 32
import tests.sweep_framework.sweeps.softmax

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

from collections import namedtuple

SoftmaxShardedMultiCoreProgramConfig = namedtuple(
    "SoftmaxShardedMultiCoreProgramConfig", ["core_x", "core_y", "subblock_w", "block_h", "block_w"]
)


def test_softmax():
    batch_sizes = (1,)
    input_a_height = 1024
    input_a_width = 1024
    input_a_dtype = ttnn.bfloat16
    device = ttnn.open_device(0)
    input_shape = (*batch_sizes, input_a_height, input_a_width)
    input_tensor = torch.randn(input_shape).bfloat16()
    torch_output_tensor = torch.softmax(input_tensor, dim=-1)
    tt_input_tensor = ttnn.from_torch(input_tensor, dtype=input_a_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    start_time = start_measuring_time()
    tt_output_tensor_on_device = ttnn.softmax(tt_input_tensor, compute_kernel_config=compute_kernel_config)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, tt_output_tensor, 0.999), e2e_perf]


def helper():
    batch_sizes = (1,)
    input_a_height = 1024
    input_a_width = 1024
    input_a_dtype = ttnn.bfloat16
    device = ttnn.open_device(0)
    input_shape = (*batch_sizes, input_a_height, input_a_width)
    input_tensor = torch.randn(input_shape).bfloat16()
    tt_input_tensor = ttnn.from_torch(input_tensor, dtype=input_a_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tests.sweep_framework.sweeps.softmax.softmax_elastic_search_check(tt_input_tensor, -1)


def plotter1():
    times = [
        236,
        450,
        600,
        837,
        1077,
        1324,
        1558,
        1790,
        2028,
        2272,
        2620,
        2748,
        2994,
        3239,
        3597,
        3722,
        3958,
        4321,
        4432,
        4676,
        5027,
        5154,
        5508,
        5636,
        5984,
        6111,
        6476,
        6614,
        6927,
        7021,
        7400,
        7643,
        7787,
        8110,
        8277,
        8481,
        8836,
        9092,
        9344,
        9458,
        9826,
        10099,
        10287,
        10527,
        10824,
        11051,
        11249,
        11483,
        11774,
        11905,
        12347,
        12495,
        12718,
        12938,
        13203,
        13448,
        13651,
        13855,
        14130,
        14395,
        14764,
        14888,
        15076,
        15409,
        16573,
        15977,
        16199,
        16503,
        16628,
        16792,
        17091,
        17288,
        17676,
        17854,
        18123,
        18257,
        18772,
        18842,
        19287,
        19236,
        19591,
        19841,
        20165,
        20278,
        20497,
        20636,
        21021,
        21231,
        21513,
        21663,
        22783,
        22282,
        22824,
        22744,
        22851,
        23201,
        23475,
        23752,
        23854,
        24185,
    ]
    index = range(1, 200, 2)
    plt.xlabel("Number of calls")
    plt.ylabel("Time [ms]")
    plt.title("Time spent on calls to elasticsearch")
    plt.plot(index, times)
    plt.savefig("python_calls.png", format="png")
    plt.close()


def plotter2():
    times = [
        118,
        356,
        593,
        831,
        1071,
        1190,
        1189,
        1191,
        1186,
        1190,
        1190,
        1199,
        1201,
        1199,
        1191,
        1200,
        1202,
        1201,
        1199,
        1211,
        1211,
        1209,
        1203,
        1205,
        1217,
        1201,
        1206,
        1203,
        1217,
        1206,
        1220,
        1211,
        1221,
        1217,
        1219,
        1214,
        1218,
        1227,
        1217,
        1215,
        1222,
        1216,
        1276,
        1239,
        1265,
        1253,
        1249,
        1253,
        1251,
        1252,
        1263,
        1253,
        1265,
        1275,
        1285,
        1282,
        1283,
        1272,
        1274,
        1287,
        1285,
        1300,
        1290,
        1285,
        1290,
        1297,
        1305,
        1316,
        1328,
        1309,
        1316,
        1331,
        1315,
        1318,
        1313,
        1325,
        1316,
        1329,
        1334,
        1336,
        1371,
        1368,
        1387,
        1362,
        1460,
        1362,
        1396,
        1375,
        1380,
        1364,
        1360,
        1355,
        1357,
        1367,
        1396,
        1337,
        1343,
        1349,
        1372,
        1321,
    ]
    index = range(1, 200, 2)
    plt.xlabel("Number of calls")
    plt.ylabel("Time [ms]")
    plt.title("Time spent on calls to elasticsearch")
    plt.plot(index, times)
    plt.savefig("cpp_calls.png", format="png")
    plt.close()


def plotter3():
    times = [
        135,
        359,
        594,
        832,
        1063,
        1182,
        1190,
        1190,
        1198,
        1193,
        1191,
        1201,
        1202,
        1201,
        1204,
        1200,
        1207,
        1196,
        1203,
        1212,
        1211,
        1216,
        1214,
        1212,
        1214,
        1215,
        1212,
        1213,
        1225,
        1227,
        1219,
        1215,
        1224,
        1223,
        1224,
        1226,
        1224,
        1226,
        1223,
        1226,
        1224,
        1223,
        1216,
        1224,
        1225,
        1225,
        1227,
        1225,
        1226,
        1225,
        1238,
        1238,
        1231,
        1237,
        1236,
        1234,
        1222,
        1224,
        1240,
        1237,
        1237,
        1238,
        1238,
        1236,
        1237,
        1259,
        1229,
        1246,
        1234,
        1259,
        1236,
        1260,
        1246,
        1259,
        1282,
        1281,
        1291,
        1284,
        1295,
        1284,
        1290,
        1284,
        1282,
        1305,
        1293,
        1294,
        1295,
        1291,
        1305,
        1289,
        1289,
        1304,
        1305,
        1291,
        1318,
        1328,
        1329,
        1326,
        1356,
        1327,
    ]
    index = range(1, 200, 2)
    plt.xlabel("Number of calls")
    plt.ylabel("Time [ms]")
    plt.title("Time spent on calls to elasticsearch")
    plt.plot(index, times)
    plt.savefig("cpp_calls_2_vectors.png", format="png")
    plt.close()


if __name__ == "__main__":
    # times = [236, 450, 600, 837, 1077, 1324, 1558, 1790, 2028, 2272, 2620, 2748, 2994, 3239, 3597, 3722, 3958, 4321, 4432, 4676, 5027, 5154, 5508, 5636, 5984, 6111, 6476, 6614, 6927, 7021, 7400, 7643, 7787, 8110, 8277, 8481, 8836, 9092, 9344, 9458, 9826, 10099, 10287, 10527, 10824, 11051, 11249, 11483, 11774, 11905, 12347, 12495, 12718, 12938, 13203, 13448, 13651, 13855, 14130, 14395, 14764, 14888, 15076, 15409, 16573, 15977, 16199, 16503, 16628, 16792, 17091, 17288, 17676, 17854, 18123, 18257, 18772, 18842, 19287, 19236, 19591, 19841, 20165, 20278, 20497, 20636, 21021, 21231, 21513, 21663, 22783, 22282, 22824, 22744, 22851, 23201, 23475, 23752, 23854, 24185]
    # index = range(1,200,2)
    # plt.xlabel('Number of calls')
    # plt.ylabel('Time [ms]')
    # plt.title('Time spent on calls to elasticsearch')
    # plt.plot(index, times)
    # plt.savefig("python_calls.png", format = 'png')
    # plt.close()
    plotter3()
