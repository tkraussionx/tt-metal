import math
import torch

import tt_lib as ttl


class Pooling:
    in_hw = (0, 0)  # in_h, in_w
    out_hw = (0, 0)
    stride_hw = (0, 0)
    pad_hw = (0, 0)
    window_hw = (0, 0)
    dilation_hw = (0, 0)

    ## given out stick range, calculate corresponding window's center stick input coords
    def calculate_in_range(self, out_range):
        ## start of the range
        out_w_i = out_range[0] % self.out_w
        out_h_i = out_range[0] / self.out_w
        in_w_i = out_w_i * self.stride_w
        in_h_i = out_h_i * self.stride_h
        in_range_start = in_h_i * self.in_w + in_w_i
        ## end of the range
        out_w_i = out_range[1] % self.out_w
        out_h_i = out_range[1] / self.out_w
        in_w_i = out_w_i * self.stride_w
        in_h_i = out_h_i * self.stride_h
        in_range_end = in_h_i * self.in_w + in_w_i
        return (in_range_start, in_range_end)

    def insert_pad(self, in_data: torch.Tensor, pad_val=0x0):
        ## padding is to be insered on h and w dims, and not c dim
        ## a.shape: [n, h, w, c]
        ## pad1 = torch.zeros([n, h, 1, c])
        ## w_padded = torch.cat((pad1, a, pad1), 2)
        ## pad2 = torch.zeros([n, 1, w + 2, c])
        ## hw_padded = torch.cat((pad2, w_padded), 1)
        ## hw_padded = torch.reshape(hw_padded, [n * (h + 1) * (w + 2), c])
        ## pad3 = torch.reshape(pad2[0], [(w + 2), c])
        ## hw_padded = torch.cat((hw_padded, pad3))
        [in_n, in_h, in_w, in_c] = in_data.shape
        ## pad for both sides along w
        pad_w = torch.ones([in_n, in_h, 1, in_c]) * pad_val
        data_w_padded = torch.cat((pad_w, in_data, pad_w), 2)
        ## pad once per batch along h
        pad_h = torch.ones([in_n, 1, in_w + 2, in_c]) * pad_val
        data_hw_padded = torch.cat((pad_h, data_w_padded), 1)
        ## one last pad along h also needed
        pad_h = torch.reshape(pad_h[0], [(in_w + 2), in_c])
        data_hw_padded = torch.cat((data_hw_padded, pad_h))
        return data_hw_padded

    def insert_halo(self, in_data: torch.Tensor, ncores, in_n):
        total_out_nsticks = self.out_hw[0] * self.out_hw[1] * in_n
        out_nsticks_per_core = int(math.ceil(float(total_out_nsticks) / ncores))
        halo_nsticks = (self.in_hw[1] + 2 * self.pad_hw[1]) * (self.window_hw[0] / 2) + (self.window_hw[1] / 2)
        out_start = 0
        for i in range(ncores):
            curr_shard = torch.tensor()
            out_end = out_start + out_nsticks_per_core
            in_start, in_end = self.calculate_in_range((out_start, out_end))
            out_start += out_nsticks_per_core
        ## ...
        return in_data

    def calc_out(self):
        out_h = (
            math.floor(
                (self.in_hw[0] + 2 * self.pad_hw[0] - (self.dilation_hw[0] * self.kernel_hw[0] - 1) - 1)
                / self.stride_hw[0]
            )
            + 1
        )
        out_w = (
            math.floor(
                (self.in_hw[1] + 2 * self.pad_hw[1] - (self.dilation_hw[1] * self.kernel_hw[1] - 1) - 1)
                / self.stride_hw[1]
            )
            + 1
        )
        return (out_h, out_w)

    def __init__(self, in_hw, stride_hw, pad_hw, window_hw, dilation_hw):
        self.in_hw = in_hw
        self.stride_hw = stride_hw
        self.pad_hw = pad_hw
        self.window_hw = window_hw
        self.dilation_hw = dilation_hw
        self.out_hw = calc_out(self)


def reshard_with_halo(
    in_tensor: ttl.tensor.Tensor,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    window_h=1,
    window_w=1,
    dilation_h=1,
    dilation_w=1,
):
    assert in_tensor.is_sharded()
    shard_spec = in_tensor.shard_spec()
    in_memory_config = in_tensor.memory_config()

    ## check for supported cases
    assert in_memory_config.memory_layout == ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    assert shard_spec.shard_orientation == ttl.tensor.ShardOrientation.ROW_MAJOR

    ncores = shard_spec.num_cores()

    in_n = in_tensor.shape()[0]
    in_c = in_tensor.shape()[3]  ## assumes channels is last

    in_h = in_tensor.shape()[1]
    in_w = in_tensor.shape()[2]

    if in_h == 1:
        # h and w are folded together, assuming in_h == in_w
        in_h = int(math.sqrt(in_w))
        in_w = in_h

    pool = Pooling(
        in_hw=(in_h, in_w),
        stride_hw=(stride_h, stride_w),
        pad_hw=(pad_h, pad_w),
        window_hw=(window_h, window_w),
        dilation_hw=(dilation_h, dilation_w),
    )

    # first convert input tensor to torch with appropriate shape
    in_data = in_tensor.to_torch().rehape([in_n, in_h, in_w, in_c])
    in_data_with_pad = pool.insert_pad(in_data)
    out_data = pool.insert_halo(in_data_with_pad, ncores, in_n)

    # now construct sharded ttl.tensor.Tensor using out_data
    out_tensor = None
    # ...

    return out_tensor
