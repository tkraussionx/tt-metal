import tt_lib as ttl

from typing import Union, List


def run_max_pool_on_device_wrapper(device, kernel_size, stride, padding, channels_last=False):
    def max_pool_2d(x):
        if channels_last:
            x = ttl.tensor.permute(x, 0, 3, 1, 2)
        out = ttl.tensor.max_pool2d(x, kernel_size, kernel_size, stride, stride, padding, padding)
        if channels_last:
            out = ttl.tensor.permute(out, 0, 2, 3, 1)
        return out

    return max_pool_2d
