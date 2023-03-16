import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from pymetal import ttmetal as ttm

def SiLU(x):
    xs = ttm.tensor.sigmoid(x)
    xs = ttm.tensor.mul(xs, x)
    return xs
