# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import sys
import json

import tt_lib as ttl
import tt_lib.fallback_ops
import tt_lib.profiler as profiler

from loguru import logger


def dealloc_input(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    for a in args:
        if type(a) is list:
            for e in a:
                if type(a) is ttl.tensor.Tensor:
                    while a.is_allocated():
                        ttnn.deallocate(a)
        if type(a) is ttl.tensor.Tensor:
            while a.is_allocated():
                ttnn.deallocate(a)
    return out


mstats = []


def write_mstats():
    print("write mstats")
    f = open("memstats.json", "w")
    f.write(json.dumps(mstats))


import atexit

atexit.register(write_mstats)


def dump(message, device, *args):
    print(message, ttl.device.GetDeviceMemoryState(device)["L1"], *args)
    d = {}
    d["message"] = message
    d["state"] = ttl.device.GetDeviceMemoryState(device)["L1"]
    mstats.append(d)


def unet_concat(ttnn_tensors, dim=-1, use_reshard=True, move=False):
    assert len(ttnn_tensors) > 0
    assert dim < 0
    rank = len(ttnn_tensors[0].shape)
    ttlib_tensors = ttnn_tensors
    all_sharded = all(t.is_sharded() for t in ttlib_tensors)
    if all_sharded:
        max_idx, output_mem_config = max(
            ((i, t.memory_config()) for i, t in enumerate(ttlib_tensors)), key=lambda m: m[1].shard_spec.num_cores()
        )
        for i in range(0, len(ttlib_tensors)):
            if i == max_idx:
                continue
            t = ttlib_tensors[i]
            t_mem_config = t.memory_config()
            t_shard_shape = t_mem_config.shard_spec.shape
            output_shard_shape = output_mem_config.shard_spec.shape
            output_shard_shape[dim] += t_shard_shape[dim]
            output_mem_config.shard_spec.shape = output_shard_shape

            reshard_shape = output_shard_shape
            reshard_shape[dim] = t_shard_shape[dim]
            if reshard_shape != t_shard_shape:
                t_mem_config.shard_spec.shape = reshard_shape
                t_mem_config.shard_spec.grid = output_mem_config.shard_spec.grid
                t_mem_config.shard_spec.orientation = output_mem_config.shard_spec.orientation
                ttlib_tensors[i] = dealloc_input(ttl.tensor.reshard, t, t_mem_config)
                dump("asdf reshard (cc)", ttlib_tensors[i].device())
                if move:
                    ttlib_tensors[i] = ttl.tensor.move_sharded(ttlib_tensors[i])
                    dump("asdf move (cc)", ttlib_tensors[i].device())
    else:
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
        for i in range(0, len(ttlib_tensors)):
            if ttlib_tensors[i].is_sharded():
                ttlib_tensors[i] = ttnn.to_memory_config(ttlib_tensors[i], output_mem_config)
                print("asdf", ttlib_tensors[i].dtype)
    return dealloc_input(ttl.tensor.concat, ttlib_tensors, dim=dim, output_mem_config=output_mem_config)


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.p1 = parameters.p1
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.p2 = parameters.p2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.p3 = parameters.p3
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
        self.p4 = parameters.p4
        self.bnc = parameters.bnc
        self.bnc_2 = parameters.bnc_2
        self.c5 = parameters.c5
        self.c5_2 = parameters.c5_2
        self.c5_3 = parameters.c5_3
        self.c6 = parameters.c6
        self.c6_2 = parameters.c6_2
        self.c6_3 = parameters.c6_3
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c8_3 = parameters.c8_3
        self.output_layer = parameters.output_layer

    def __call__(self, device, input_tensor, orig_shape):
        nhw = orig_shape[-4] * orig_shape[-2] * orig_shape[-1]
        # Preallocate the output
        dump("asdf begin", device, input_tensor.shape)
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)
        dump("asdf input", device, input_tensor.dtype)

        output_tensor = self.c1(input_tensor)
        dump("asdf c1", device)
        output_tensor = self.c1_2(output_tensor)
        dump("asdf c1_2", device, output_tensor.shape)
        save_c1_2_out = output_tensor
        dump("asdf c1_2 layout", device)
        output_tensor = self.p1(output_tensor)
        dump("asdf p1", device)

        output_tensor = dealloc_input(ttl.tensor.reshard, output_tensor, self.c2.conv.input_sharded_memory_config)
        dump("asdf reshard", device)
        output_tensor = dealloc_input(self.c2, output_tensor)
        dump("asdf c2", device)
        output_tensor = dealloc_input(self.c2_2, output_tensor)
        dump("asdf c2_2", device)
        save_c2_2_out = output_tensor
        dump("asdf c2_2 layout", device)
        output_tensor = self.p2(output_tensor)
        dump("asdf p2", device)

        output_tensor = dealloc_input(ttl.tensor.reshard, output_tensor, self.c3.conv.input_sharded_memory_config)
        dump("asdf reshard", device)
        output_tensor = dealloc_input(self.c3, output_tensor)
        dump("asdf c3", device)
        output_tensor = dealloc_input(self.c3_2, output_tensor)
        dump("asdf c3_2", device)
        save_c3_2_out = output_tensor
        dump("asdf c3_2 layout", device)
        output_tensor = self.p3(output_tensor)
        dump("asdf p3", device)

        output_tensor = dealloc_input(ttl.tensor.reshard, output_tensor, self.c4.conv.input_sharded_memory_config)
        dump("asdf reshard", device)
        output_tensor = dealloc_input(self.c4, output_tensor)
        dump("asdf c4", device)
        output_tensor = dealloc_input(self.c4_2, output_tensor)
        dump("asdf c4_2", device)
        save_c4_2_out = output_tensor
        output_tensor = self.p4(output_tensor)
        dump("asdf p4", device)

        output_tensor = dealloc_input(self.bnc, output_tensor)
        dump("asdf bnc", device)
        output_tensor = dealloc_input(self.bnc_2, output_tensor)
        dump("asdf bnc_2", device)

        ## upsample block
        output_tensor = dealloc_input(ttnn.to_layout, output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        dump("asdf bnc_2 layout", device)
        output_tensor = ttnn.reshape(
            output_tensor, (orig_shape[-4], orig_shape[-2] // 16, orig_shape[-1] // 16, output_tensor.shape[-1])
        )
        output_tensor = dealloc_input(ttnn.upsample, output_tensor, 2)
        dump("asdf upsample0", device)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 64, output_tensor.shape[-1]))

        save_c4_2_out = ttnn.to_layout(save_c4_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c4_2_out], dim=-1)
        while save_c4_2_out.is_allocated():
            ttnn.deallocate(save_c4_2_out)
        dump("asdf concat0", device)

        output_tensor = dealloc_input(ttl.tensor.reshard, output_tensor, self.c5.conv.input_sharded_memory_config)
        dump("asdf reshard", device)
        output_tensor = dealloc_input(self.c5, output_tensor)
        dump("asdf c5", device)
        output_tensor = dealloc_input(self.c5_2, output_tensor)
        dump("asdf c5_2", device)
        output_tensor = dealloc_input(self.c5_3, output_tensor)
        dump("asdf c5_3", device)

        output_tensor = dealloc_input(ttnn.to_layout, output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        dump("asdf c5_3 layout", device)
        output_tensor = ttnn.reshape(
            output_tensor, (orig_shape[-4], orig_shape[-2] // 8, orig_shape[-1] // 8, output_tensor.shape[-1])
        )
        output_tensor = dealloc_input(ttnn.upsample, output_tensor, 2)
        dump("asdf upsample1", device)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 16, output_tensor.shape[-1]))

        save_c3_2_out = ttnn.to_layout(save_c3_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c3_2_out], dim=-1)
        while save_c3_2_out.is_allocated():
            ttnn.deallocate(save_c3_2_out)
        dump("asdf concat1", device)

        output_tensor = dealloc_input(ttl.tensor.reshard, output_tensor, self.c6.conv.input_sharded_memory_config)
        dump("asdf reshard", device)
        output_tensor = dealloc_input(self.c6, output_tensor)
        dump("asdf c6", device)
        output_tensor = dealloc_input(self.c6_2, output_tensor)
        dump("asdf c6_2", device)
        output_tensor = dealloc_input(self.c6_3, output_tensor)
        dump("asdf c6_3", device)

        output_tensor = dealloc_input(ttnn.to_layout, output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        dump("asdf layout", device)
        output_tensor = ttnn.reshape(
            output_tensor, (orig_shape[-4], orig_shape[-2] // 4, orig_shape[-1] // 4, output_tensor.shape[-1])
        )
        output_tensor = dealloc_input(ttnn.upsample, output_tensor, 2)
        dump("asdf upsample2", device)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 4, output_tensor.shape[-1]))

        save_c2_2_out = ttnn.to_layout(save_c2_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c2_2_out], dim=-1)
        while save_c2_2_out.is_allocated():
            ttnn.deallocate(save_c2_2_out)
        dump("asdf concat2", device)

        hacked_shard_shape = self.c7.conv.input_sharded_memory_config.shard_spec.shape
        hacked_shard_shape[1] = output_tensor.shape[-1]
        self.c7.conv.input_sharded_memory_config.shard_spec.shape = hacked_shard_shape
        output_tensor = dealloc_input(self.c7, output_tensor)
        dump("asdf c7", device)
        output_tensor = dealloc_input(self.c7_2, output_tensor)
        dump("asdf c7_2", device)
        output_tensor = dealloc_input(self.c7_3, output_tensor)
        dump("asdf c7_3", device)

        # output_tensor = dealloc_input(ttnn.to_layout, output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        # dump("asdf c7_3 layout", device)
        output_tensor = ttnn.reshape(
            output_tensor, (orig_shape[-4], orig_shape[-2] // 2, orig_shape[-1] // 2, output_tensor.shape[-1])
        )
        output_tensor = dealloc_input(ttnn.upsample, output_tensor, 2)
        dump("asdf upsample3", device)
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw, output_tensor.shape[-1]))

        save_c1_2_out = dealloc_input(ttnn.to_layout, save_c1_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        dump("asdf to_layout", device)
        save_c1_2_out = ttl.tensor.move_sharded(save_c1_2_out)
        dump("asdf move save_c1_2_out", device)
        output_tensor = ttl.tensor.move_sharded(output_tensor)
        dump("asdf move output_tensor", device)
        output_tensor = unet_concat([output_tensor, save_c1_2_out], dim=-1, move=True)
        while save_c1_2_out.is_allocated():
            ttnn.deallocate(save_c1_2_out)
        dump("asdf concat3", device)
        # output_tensor = dealloc_input(ttnn.to_layout, output_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        # dump("asdf tilize", device)
        output_tensor = ttl.tensor.move_sharded(output_tensor)
        dump("asdf move", device)

        output_tensor = dealloc_input(self.c8, output_tensor)
        dump("asdf c8", device)
        output_tensor = dealloc_input(self.c8_2, output_tensor)
        dump("asdf c8_2", device)
        output_tensor = dealloc_input(self.c8_3, output_tensor)
        dump("asdf c8_3", device)
        output_tensor = dealloc_input(self.output_layer, output_tensor)
        dump("asdf output", device)
        return ttnn.from_device(output_tensor)
