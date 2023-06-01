import pytest

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/..")

import numpy as np
from loguru import logger

import tt_lib

from python_api_testing.sweep_tests.generation_funcs import gen_rand_multi

import torch

size = [[1,1,32,32]]
low=[[10,100, 200] ]
high=[[20,110, 210]]

@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)

def test_multi_ranges(size, low, high):
    logger.info("Running value range test")
    logger.info(f'Size: {size}')
    logger.info(f'Low: {low}')
    logger.info(f'High: {high}')

    total = 1
    for j in range(len(size)):
      total*=size[j]

    t1= gen_rand_multi(size, low, high)
    t1 = t1.reshape(total)

    fails = False
    for t in t1:
      fails = True
      for i in range(len(low)):
        if(t<low[i] or t>high[i]):
          fails = fails and True
        else:
          fails = fails and False
      if(fails):
        break
    assert(not(fails))
    logger.info("Range test  passed")


@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)
def test_shape(size, low, high):
    logger.info("Running shape check test")
    logger.info(f'Size: {size}')
    logger.info(f'Low: {low}')
    logger.info(f'High: {high}')
    t1= gen_rand_multi(size, low, high)
    assert(t1.shape == torch.Size(size))
    logger.info("Shape test passed")

if __name__ == "__main__":
    test_shape(size, low, high)
    test_multi_ranges(size, low, high)
