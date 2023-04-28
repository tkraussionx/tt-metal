from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    channels_last,
    _nearest_32,
    convert_weights_2d_matrix,
    convert_act_2d_matrix,
)

import torch
import random
import numpy

def run_rw_test(shape, num_blocks, block_size, read_size):

    def generate_random_address_map():
        src_address = 0
        dst_address = 0
        pad = 0
        num_reads_per_block = (int) (block_size / read_size)
        address_map = []
        for i in range(num_blocks):
            address_map.append(num_reads_per_block)
            for j in range(num_reads_per_block):
                # generate random src address for scattered reads
                #src_address = random.randint(0, numpy.prod(shape))
                #pad = random.randint(0,1)
                address_map.append(src_address*2) # *2 for bytes
                address_map.append(dst_address*2)
                address_map.append(read_size*2)
                address_map.append(pad)
                src_address += read_size
                dst_address += read_size
        return address_map
    address_map = generate_random_address_map()

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    A_pyt = torch.randn(shape, dtype=torch.bfloat16).float()
    A = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
        ttl.tensor.MemoryConfig(False, 0), # single bank
    )
    A_t = ttl.tensor.reader_writer_op_multi_blocks(A, address_map, num_blocks, block_size)
    ttl.device.CloseDevice(device)

def test_run_rw_blocked():
    # Tests that pass -
    #run_rw_test([1,1,1*32,1*32], 1, 1*1*1024, 32)
    #run_rw_test([1,1,2*32,2*32], 1, 2*2*1024, 32)
    #run_rw_test([1,1,4*32,4*32], 2, 4*2*1024, 32)
    #run_rw_test([1,1,7*32,36*32], 12, 7*3*1024, 96)

    # Crashes -
    run_rw_test([1,1,7*32,72*32], 24, 7*3*1024, 96)
