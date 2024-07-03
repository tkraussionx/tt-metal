from loguru import logger
from itertools import permutations
import random
import pytest
import torch
import tt_lib as ttl
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_addcmul


data_seeds = [16305027, 7329721, 3181992, 4931206, 11079580]
scalars = [83.0, 7.8125, 95.5, 8.625, -8.625]
test_sweep_args = []
for i, shape in enumerate([(4, 7, 32, 96), (4, 5, 128, 96), (5, 3, 96, 64), (5, 5, 160, 192), (3, 2, 192, 32)]):
    for in_mem_cfg_ in list(
        permutations(
            [
                None,
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ]
        )
    ):
        if in_mem_cfg_[0] is None or in_mem_cfg_[1] is None:
            continue

        test_sweep_args.append(
            (
                shape,
                [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
                [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
                list(in_mem_cfg_),
                ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
                data_seeds[i],
                scalars[i],
            )
        )


print(len(test_sweep_args))
