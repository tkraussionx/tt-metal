# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def trace_sliding_window_op_to_generate_data_top_left_indices_and_pad_metadata(
    sliding_window_op_params, input_nchw_shape
):
    assert len(sliding_window_op_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        sliding_window_op_params[i] for i in range(10)
    ]
    assert dilation == 1 and groups == 1
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    # image 1 data
    # 1  2  3  4  5  6  7  8
    # 9  10 11 12 13 14 15 16
    # 17 18 19 20 21 22 23 24
    # 25 26 27 28 29 30 31 32
    # image 2 data
    # 33 34 35 36 37 38 39 40
    # 41 42 43 44 45 46 47 48
    # 49 50 51 52 53 54 55 56
    # 57 58 59 60 61 62 63 64

    # Concatenated image data from above
    # Inserted padding above and between and on the sides of the images (pad = 1)
    # 0  0  0  0  0  0  0  0  0 0
    # 0  1  2  3  4  5  6  7  8 0
    # 0  9 10 11 12 13 14 15 16 0
    # 0 17 18 19 20 21 22 23 24 0
    # 0 25 26 27 28 29 30 31 32 0
    # 0  0  0  0  0  0  0  0  0 0
    # 0  0  0  0  0  0  0  0  0 0
    # 0 33 34 35 36 37 38 39 40 0
    # 0 41 42 43 44 45 46 47 48 0
    # 0 49 50 51 52 53 54 55 56 0
    # 0 57 58 59 60 61 62 63 64 0
    # 0  0  0  0  0  0  0  0  0 0

    # We encode above shown padded tensor into pad_metadata (list of boolean - true if padding location)
    # pad_meta_data: [true, true, ..., false, ...]

    padded_input_h = input_h + (2 * pad_h)
    padded_input_w = input_w + (2 * pad_w)
    pad_metadata = []
    for n in range(input_n):
        for h in range(padded_input_h):
            for w in range(padded_input_w):
                if h < pad_h or h >= (input_h + pad_h) or w < pad_w or w >= (input_w + pad_w):
                    pad_metadata.append(True)
                else:
                    pad_metadata.append(False)

    # TODO: add support for dilation > 1
    output_h = ((int)((padded_input_h - filter_h) / stride_h)) + 1
    output_w = ((int)((padded_input_w - filter_w) / stride_w)) + 1
    # generate a list of input indices corresponding to the top left position of sliding window
    # the index refers to the location in the padded tensor
    data_top_left_indices = []
    for n in range(input_n):
        for oh in range(output_h):
            for ow in range(output_w):
                ih = oh * stride_h
                iw = ow * stride_w
                channel_idx = (n * padded_input_h * padded_input_w) + (ih * padded_input_w) + iw
                data_top_left_indices.append(channel_idx)

    return pad_metadata, data_top_left_indices


def decompose_sliding_window_op_into_shards_and_generate_tensor_metadata(
    data_top_left_indices,
    pad_metadata,
    input_padded_w,
    sldw_op_output_shard_height,
    unpadded_input_shard_height,
    num_cores,
    filter_h,
    filter_w,
):
    req_sldw_op_input_shard_start_end = []  # start and end indices refer to global padded input tensor
    sldw_op_output_start_stick = 0
    for core_id in range(num_cores):
        if sldw_op_output_start_stick >= len(data_top_left_indices):
            print("core_id=", core_id)
            print("sldw_op_output_start_stick=", sldw_op_output_start_stick)
            print("len(data_top_left_indices)=", len(data_top_left_indices))
            print("sldw_op_output_shard_height=", sldw_op_output_shard_height)
        assert sldw_op_output_start_stick < len(data_top_left_indices)
        req_sldw_op_input_shard_start_stick = data_top_left_indices[sldw_op_output_start_stick]
        sldw_op_output_end_stick = (
            min(sldw_op_output_start_stick + sldw_op_output_shard_height, len(data_top_left_indices)) - 1
        )
        req_sldw_op_input_shard_end_stick = data_top_left_indices[sldw_op_output_end_stick]
        halo_with_pad_nsticks = ((filter_h - 1) * input_padded_w) + filter_w - 1
        req_sldw_op_input_shard_end_stick += halo_with_pad_nsticks
        req_sldw_op_input_shard_start_end.append(
            (
                (sldw_op_output_start_stick, sldw_op_output_end_stick),
                (req_sldw_op_input_shard_start_stick, req_sldw_op_input_shard_end_stick),
            )
        )
        sldw_op_output_start_stick += sldw_op_output_shard_height

    tensor_metadata = []
    unpadded_input_shard_local_idx = 0
    core_id = 0
    for padded_input_tensor_idx in range(len(pad_metadata)):
        pad_stick = pad_metadata[padded_input_tensor_idx]
        if pad_stick:
            tensor_metadata.append((pad_stick, 0, 0))
        else:
            # sanity check
            assert core_id < num_cores
            assert unpadded_input_shard_local_idx < unpadded_input_shard_height
            tensor_metadata.append((pad_stick, core_id, unpadded_input_shard_local_idx))
            unpadded_input_shard_local_idx += 1
            if unpadded_input_shard_local_idx == unpadded_input_shard_height:
                unpadded_input_shard_local_idx = 0
                core_id += 1
    assert len(tensor_metadata) == len(pad_metadata)
    return req_sldw_op_input_shard_start_end, tensor_metadata


NEIGHBORHOOD_DIST = 2  ## ll, l, r, rr


## Function to generate the untilize with halo writer kernel config using the tensor metadata and required shard start/end information.
##
## Inputs:  1. tensor_metadata:             [(is_pad, src_core_id, src_local_idx), ...], size = padded tensor size
##                                              NOTE: (src_core_id, src_local_idx) == src_global_idx
##          2. resharded_start_and_end:     [(req_shard_start, req_shard_end), ...], size = num cores
##
## Outputs: 1. local_data_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##          2. local_pad_start_and_size:    [[(dst_start, size), ...], ...], size = num cores
##          3. neighbor data config:            NOTE: currently NEIGHBORHOOD_DIST = 2. Can be generalized.
##              1. ll_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
##              2. l_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              3. r_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              4. rr_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
def generate_untilize_with_halo_kernel_configs(tensor_metadata: list, resharded_start_and_end: list):
    # print(f'tensor metadata: {tensor_metadata}')
    ncores = len(resharded_start_and_end)

    ## data :: { core -> [
    ##              [],    ## ll
    ##              [],    ## l
    ##              [],    ## local
    ##              [],    ## r
    ##              [],    ## rr
    ##          ]}
    core_neighbor_data = {}
    core_pad_start_and_size = {}
    core_src_local_start_idx = {}  ## {core -> [ ll, l, local, r, rr ]}

    ## NOTE: assuming the core_id's are contiguous
    for dst_core_id in np.arange(ncores):
        ## generate the config for dst_core_id using the input metadata

        dst_global_start_idx, dst_global_end_idx = resharded_start_and_end[dst_core_id][1]

        core_pad_start_and_size[dst_core_id] = []

        curr_segment_size = 0
        is_curr_segment_pad = None
        curr_segment_src_core_id = None
        curr_segment_dst_start_idx = None
        curr_segment_neighbor_idx = None

        for dst_global_idx in np.arange(dst_global_start_idx, dst_global_end_idx + 1):
            dst_local_idx = dst_global_idx - dst_global_start_idx
            is_pad, src_core_id, src_local_idx = tensor_metadata[dst_global_idx]

            if is_pad:  ## pad stick
                if curr_segment_size > 0 and is_curr_segment_pad:
                    ## current segment is padding
                    curr_segment_size += 1
                else:
                    if curr_segment_size > 0:
                        ## current segment is data, a new pad segment starts here
                        ## finish off the data seg first
                        if curr_segment_src_core_id not in core_neighbor_data:
                            core_neighbor_data[curr_segment_src_core_id] = []
                            for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                core_neighbor_data[curr_segment_src_core_id].append([])
                        core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                            (curr_segment_dst_start_idx, curr_segment_size)
                        )
                    else:
                        ## there is no current segment
                        pass
                    ## start new pad segment
                    is_curr_segment_pad = True
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx

            else:  ## data stick
                ## the neighbor core of dst_core_id this data stick is coming from (src_core_id): ll, l, local, r or rr
                neighbor_idx = NEIGHBORHOOD_DIST + (dst_core_id - src_core_id)
                assert neighbor_idx >= 0 and neighbor_idx < 2 * NEIGHBORHOOD_DIST + 1

                if curr_segment_size > 0:
                    if curr_segment_src_core_id == src_core_id:
                        ## this data stick belong to the same src core as current segment
                        ## if the curr segment is also data, then it is contiguous
                        ## else, this is new data segment after a pad break
                        if not is_curr_segment_pad:
                            ## contiguous data stick
                            curr_segment_size += 1
                        else:
                            ## curr segment is padding, and a new data segment starts here
                            ## finish off the pad segment first (always local only)
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                            ## start the new data segment
                            is_curr_segment_pad = False
                            curr_segment_size = 1
                            curr_segment_dst_start_idx = dst_local_idx
                            curr_segment_src_core_id = src_core_id
                            curr_segment_neighbor_idx = neighbor_idx
                            if curr_segment_src_core_id not in core_src_local_start_idx:
                                core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                            if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                                core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx
                    else:
                        if not is_curr_segment_pad:
                            ## this data stick belongs to a different src core than the current data segment
                            ## first finish the current data segment
                            if curr_segment_src_core_id not in core_neighbor_data:
                                core_neighbor_data[curr_segment_src_core_id] = []
                                for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                    core_neighbor_data[curr_segment_src_core_id].append([])
                            core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                                (curr_segment_dst_start_idx, curr_segment_size)
                            )
                        else:
                            ## current segment is padding, finish it off
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                        ## start the new data segment
                        is_curr_segment_pad = False
                        curr_segment_size = 1
                        curr_segment_dst_start_idx = dst_local_idx
                        curr_segment_src_core_id = src_core_id
                        curr_segment_neighbor_idx = neighbor_idx
                        if curr_segment_src_core_id not in core_src_local_start_idx:
                            core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                        if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                            core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx
                else:
                    ## there is no current segment, create new data segment
                    is_curr_segment_pad = False
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx
                    curr_segment_src_core_id = src_core_id
                    curr_segment_neighbor_idx = neighbor_idx
                    if curr_segment_src_core_id not in core_src_local_start_idx:
                        core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                    if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                        core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx

        ## finish off the remaining last segment, if any
        if curr_segment_size > 0:
            if is_curr_segment_pad:
                ## padding segment
                core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
            else:
                ## data segment
                if curr_segment_src_core_id not in core_neighbor_data:
                    core_neighbor_data[curr_segment_src_core_id] = []
                    for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                        core_neighbor_data[curr_segment_src_core_id].append([])
                core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                    (curr_segment_dst_start_idx, curr_segment_size)
                )
    ll_data_start_and_size = []
    l_data_start_and_size = []
    local_data_start_and_size = []
    r_data_start_and_size = []
    rr_data_start_and_size = []
    local_pad_start_and_size = []
    src_local_start_idx = []
    local_pad_nsegments_per_core = []
    ll_data_nsegments_per_core = []
    l_data_nsegments_per_core = []
    local_data_nsegments_per_core = []
    r_data_nsegments_per_core = []
    rr_data_nsegments_per_core = []
    max_ll_data_nsegments_across_cores = 0
    max_l_data_nsegments_across_cores = 0
    max_local_data_nsegments_across_cores = 0
    max_r_data_nsegments_across_cores = 0
    max_rr_data_nsegments_across_cores = 0
    max_local_pad_nsegments_across_cores = 0

    for i in range(ncores):
        ll_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2])
        ll_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2]))
        max_ll_data_nsegments_across_cores = max(
            max_ll_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2])
        )

        l_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1])
        l_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1]))
        max_l_data_nsegments_across_cores = max(
            max_l_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1])
        )

        local_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST])
        local_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST]))
        max_local_data_nsegments_across_cores = max(
            max_local_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST])
        )

        r_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1])
        r_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1]))
        max_r_data_nsegments_across_cores = max(
            max_r_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1])
        )

        rr_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2])
        rr_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2]))
        max_rr_data_nsegments_across_cores = max(
            max_rr_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2])
        )

        local_pad_start_and_size.append(core_pad_start_and_size[i])
        local_pad_nsegments_per_core.append(len(core_pad_start_and_size[i]))
        max_local_pad_nsegments_across_cores = max(
            max_local_pad_nsegments_across_cores, len(core_pad_start_and_size[i])
        )

        # print(f'{core_src_local_start_idx[i]}')
        src_local_start_idx.append(core_src_local_start_idx[i])

    # Pad all config arrays to max nsegments since it needs to be sharded equally across cores
    # Also, flatten the list of tuples
    for i in range(ncores):
        ll_data_start_and_size[i].extend(
            [(0, 0)] * (max_ll_data_nsegments_across_cores - ll_data_nsegments_per_core[i])
        )
        ll_data_start_and_size[i] = [item for tuple_item in ll_data_start_and_size[i] for item in tuple_item]
        l_data_start_and_size[i].extend([(0, 0)] * (max_l_data_nsegments_across_cores - l_data_nsegments_per_core[i]))
        l_data_start_and_size[i] = [item for tuple_item in l_data_start_and_size[i] for item in tuple_item]
        local_data_start_and_size[i].extend(
            [(0, 0)] * (max_local_data_nsegments_across_cores - local_data_nsegments_per_core[i])
        )
        local_data_start_and_size[i] = [item for tuple_item in local_data_start_and_size[i] for item in tuple_item]
        r_data_start_and_size[i].extend([(0, 0)] * (max_r_data_nsegments_across_cores - r_data_nsegments_per_core[i]))
        r_data_start_and_size[i] = [item for tuple_item in r_data_start_and_size[i] for item in tuple_item]
        rr_data_start_and_size[i].extend(
            [(0, 0)] * (max_rr_data_nsegments_across_cores - rr_data_nsegments_per_core[i])
        )
        rr_data_start_and_size[i] = [item for tuple_item in rr_data_start_and_size[i] for item in tuple_item]
        local_pad_start_and_size[i].extend(
            [(0, 0)] * (max_local_pad_nsegments_across_cores - local_pad_nsegments_per_core[i])
        )
        local_pad_start_and_size[i] = [item for tuple_item in local_pad_start_and_size[i] for item in tuple_item]

    # for core_id in range(ncores):
    #     print(f'Core {core_id}: {resharded_start_and_end[core_id][1][1] - resharded_start_and_end[core_id][1][0] + 1}')

    max_out_nsticks_per_core = max(
        [
            resharded_start_and_end[core_id][1][1] - resharded_start_and_end[core_id][1][0] + 1
            for core_id in range(ncores)
        ]
    )
    return (
        local_data_start_and_size,
        local_pad_start_and_size,
        ll_data_start_and_size,
        l_data_start_and_size,
        r_data_start_and_size,
        rr_data_start_and_size,
        src_local_start_idx,
        local_data_nsegments_per_core,
        local_pad_nsegments_per_core,
        ll_data_nsegments_per_core,
        l_data_nsegments_per_core,
        r_data_nsegments_per_core,
        rr_data_nsegments_per_core,
        max_out_nsticks_per_core,
    )
