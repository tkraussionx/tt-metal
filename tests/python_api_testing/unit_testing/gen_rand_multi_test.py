import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import pytest


def gen_rand_multi(size, low=[0, 100], high=[100,1000]):
  interval = []
  total = 1
  for j in range(len(size)):
    total*=size[j]
  print(total)
  for i in range(len(low)):
     interval.append(high[i]-low[i])
  percentage=[]
  count = 0
  for j in range(len(interval)):
    count+=interval[j]
  print(count)
  for i in range(len(interval)):
      percentage.append(interval[i]/count)
      print(percentage[i])
  fractions=[]
  tensors=[]
  for i in range(len(interval)):

    if(i==len(interval)):
      all_except_last = 0
      for i in range(len(interval)-1):
        all_except_last+= fractions[i]
      fractions.append(total-all_except_last)
    else:
      fractions.append(round(percentage[i]*total))
    print(fractions[i])
    t =(low[i]-high[i])*torch.rand(fractions[i]) + high[i]

    tensors.append(t)


  single_tensor = tensors[0]
  for i in range(1, len(tensors)):
    single_tensor = torch.cat((single_tensor, tensors[i]),0)
  t = single_tensor
  idx = torch.randperm(t.shape[0])
  t = t[idx].view(t.size())

  result = t.reshape(size)
  return result


def rand_multi_shape():
    t1= gen_rand_multi([1,1,32,32], low=[-100,-1000,-10], high=[-10, -100, -1])
    print(t1.shape)
    print(t1)
    assert(t1.shape == torch.Size([1,1,32,32]))


    t1= gen_rand_multi([1,1,512,512], low=[-100,-1000,-10,1], high=[-10, -100, -1,50])
    print(t1.shape)
    print(t1)
    assert(t1.shape == torch.Size([1,1,512,512]))
    #t1= gen_rand_multi([1,1,33,33], low=[-100,-1000,-10], high=[-10, -100, -1])

    #assert(t1.shape() == [1,1,33,])


if __name__ == "__main__":
    rand_multi_shape()
