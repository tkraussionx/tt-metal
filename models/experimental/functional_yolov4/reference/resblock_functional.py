import torch
import torch.nn as nn
import torch.nn.functional as F


def resblock(x, ch, nblocks=1, shortcut=True):
    for _ in range(nblocks):
        h = x
        h = F.conv2d(h, weight=torch.randn(ch, ch, 1, 1), bias=None, stride=1, padding=0)
        h = F.batch_norm(
            h, running_mean=torch.zeros(ch), running_var=torch.ones(ch), weight=None, bias=None, training=True
        )
        h = F.relu(h, inplace=True)
        h = F.conv2d(h, weight=torch.randn(ch, ch, 3, 3), bias=None, stride=1, padding=1)
        h = F.batch_norm(
            h, running_mean=torch.zeros(ch), running_var=torch.ones(ch), weight=None, bias=None, training=True
        )
        h = F.relu(h, inplace=True)
        x = x + h if shortcut else h
    return x


# Example usage
x = torch.randn(1, 64, 32, 32)  # Example input
ch = 64
nblocks = 3
output = resblock(x, ch, nblocks, shortcut=True)
print(output.shape)
