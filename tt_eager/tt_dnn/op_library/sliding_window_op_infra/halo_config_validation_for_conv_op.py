# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def construct_2d_padded_tensor_list(input_tensor, input_nchw_shape, pad_metadata, pad_val: torch.int16 = 0x0):
    if pad_val == 0xF7FF:
        pad_val = -1.03e34  ## TODO: how to do this in python properly???
    # Construct the padded tensor using pad_metadata
    input_padded_tensor = []
    input_tensor_idx = 0
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    # Permute input tensor from nchw shape to nhwc shape
    input_tensor_nchw = np.reshape(input_tensor, input_nchw_shape)
    input_tensor_nhwc = np.transpose(input_tensor_nchw, (0, 2, 3, 1))
    input_tensor_nhwc = np.reshape(input_tensor_nhwc, (np.prod(input_nchw_shape)))
    for i in range(len(pad_metadata)):
        for c in range(input_c):
            if pad_metadata[i]:
                input_padded_tensor.append(pad_val)
            else:
                assert input_tensor_idx < len(input_tensor_nhwc)
                input_padded_tensor.append(input_tensor_nhwc[input_tensor_idx])
                input_tensor_idx += 1
    return input_padded_tensor


def construct_input_padded_tensor(input_pyt_tensor, pad_metadata, pad_val: torch.int16 = 0x0):
    return construct_2d_padded_tensor_list(
        input_pyt_tensor.reshape(-1).tolist(), list(input_pyt_tensor.size()), pad_metadata, pad_val
    )


def validate_input_padded_tensor_and_data_top_left_indices_and_pad_metadata(
    input_padded_tensor,
    input_nchw_shape,
    pad_h,
    pad_w,
    filter_pyt_tensor,
    out_golden_pyt_tensor,
    pad_metadata,
    data_top_left_indices,
):
    input_n, input_c, input_h, input_w = input_nchw_shape
    filter_k, filter_c, filter_h, filter_w = list(filter_pyt_tensor.size())
    assert input_c == filter_c

    # permute filter tensor to be channels last - kchw --> khwc
    filter_pyt_tensor_khwc = torch.permute(filter_pyt_tensor, (0, 2, 3, 1))

    input_padded_width = input_w + (2 * pad_w)
    input_padded_height = input_h + (2 * pad_h)
    input_padded_volume = input_n * input_c * input_padded_height * input_padded_width
    assert len(input_padded_tensor) == input_padded_volume
    input_padded_pyt_tensor_nhwc = torch.tensor(input_padded_tensor).reshape(
        [input_n * input_padded_height, input_padded_width, input_c]
    )
    output_tensor = []
    # run conv over padded tensor using data_top_left_indices
    for k in range(filter_k):
        for i in data_top_left_indices:
            i_bh = (int)(i / input_padded_width)
            i_w = (int)(i % input_padded_width)
            output_tensor.append(
                torch.dot(
                    input_padded_pyt_tensor_nhwc[i_bh : i_bh + filter_h, i_w : i_w + filter_w, :].reshape(-1),
                    filter_pyt_tensor_khwc[k, :, :, :].reshape(-1),
                )
            )

    output_pyt_tensor = torch.tensor(output_tensor)
    assert np.prod(output_pyt_tensor.size()) == np.prod(out_golden_pyt_tensor.size())
    # permute output golden pytorch tensor from nchw to cnhw shape
    out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
    # compare to pytorch
    passing_pcc, output_pcc = comp_equal(out_golden_pyt_tensor_cnhw.reshape(-1), output_pyt_tensor.reshape(-1))
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc


def construct_utwh_output_shards(
    # Padded input tensor
    input_padded_tensor,
    # Padded input tensor shape
    input_nchw_padded_shape,
    # config to construct shards
    req_conv_input_shard_start_end,
):
    # reshape input padded tensor to 2d shape - [nhw, c]
    assert len(input_nchw_padded_shape) == 4
    input_n, input_c, input_padded_height, input_padded_width = [input_nchw_padded_shape[i] for i in range(4)]
    input_2d_padded_tensor = np.reshape(
        input_padded_tensor, (input_n * input_padded_height * input_padded_width, input_c)
    )
    utwh_output_shards = []
    for item in req_conv_input_shard_start_end:
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        req_conv_input_shard_size = req_conv_input_shard_end - req_conv_input_shard_start + 1
        assert req_conv_input_shard_size <= 65535  # max uint16 value
        utwh_output_shards.append(input_2d_padded_tensor[req_conv_input_shard_start : req_conv_input_shard_end + 1, :])
    return utwh_output_shards


def validate_utwh_output_shards_and_req_conv_input_shard_start_end(
    # Padded input tensor shape
    input_nchw_padded_shape,
    # Filter pytorch tensor
    filter_pyt_tensor,
    # Conv golden output tensor to compare against
    out_golden_pyt_tensor,
    # Input indices corresponding to top left position of sliding window. Used to perform conv operation.
    data_top_left_indices,
    # validate utwh output shards
    utwh_output_shards,
    # Validate this config -
    req_conv_input_shard_start_end,
):
    filter_k = filter_pyt_tensor.size()[0]
    filter_c = filter_pyt_tensor.size()[1]
    filter_h = filter_pyt_tensor.size()[2]
    filter_w = filter_pyt_tensor.size()[3]

    output_n = out_golden_pyt_tensor.size()[0]
    output_c = out_golden_pyt_tensor.size()[1]
    output_h = out_golden_pyt_tensor.size()[2]
    output_w = out_golden_pyt_tensor.size()[3]
    assert len(data_top_left_indices) == output_n * output_h * output_w
    assert len(input_nchw_padded_shape) == 4
    input_n, input_c, input_padded_height, input_padded_width = [input_nchw_padded_shape[i] for i in range(4)]
    assert filter_c == input_c
    assert output_n == input_n
    assert output_c == filter_k

    # permute filter tensor to be channels last - kchw --> khwc
    filter_pyt_tensor_khwc = torch.permute(filter_pyt_tensor, (0, 2, 3, 1))

    # Perform conv on input shards one at a time, and compare against output. Use data_top_left_indices (global) to perform the conv operation.
    output_stick_global = 0
    for input_shard_idx, item in enumerate(req_conv_input_shard_start_end):
        assert input_shard_idx < len(utwh_output_shards)
        conv_output_shard_start, conv_output_shard_end = item[0]
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        # sanity check that the first item in the shard is at the top left position of sliding window
        assert output_stick_global < len(data_top_left_indices)
        assert req_conv_input_shard_start == data_top_left_indices[output_stick_global]
        output_shard = []
        output_shard_size = conv_output_shard_end - conv_output_shard_start + 1
        for k in range(filter_k):
            output_stick = output_stick_global
            for o in range(output_shard_size):
                assert output_stick < len(data_top_left_indices)
                input_top_left_position_stick = data_top_left_indices[output_stick]
                assert input_top_left_position_stick >= req_conv_input_shard_start
                input_shard_stick_local_idx = input_top_left_position_stick - req_conv_input_shard_start
                conv_input_window = []
                for fh in range(filter_h):
                    for fw in range(filter_w):
                        assert input_shard_stick_local_idx + fw < len(utwh_output_shards[input_shard_idx])
                        conv_input_window.append(
                            utwh_output_shards[input_shard_idx][input_shard_stick_local_idx + fw, :]
                        )
                    input_shard_stick_local_idx += input_padded_width
                output_val = np.dot(
                    np.array(conv_input_window).flatten(), filter_pyt_tensor_khwc[k, :, :, :].reshape(-1).tolist()
                )
                output_shard.append(output_val)
                output_stick += 1
        output_stick_global = output_stick
        output_pyt_shard = torch.tensor(output_shard).reshape((filter_k, output_shard_size))
        # compare output shard with golden output pytorch tensor
        # permute output golden pytorch tensor from nchw to cnhw shape
        out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
        # reshape cnhw to 2d shape = [c, nhw]
        out_golden_pyt_tensor_cnhw = torch.reshape(
            out_golden_pyt_tensor_cnhw, (output_c, output_n * output_h * output_w)
        )
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor_cnhw[:, conv_output_shard_start : conv_output_shard_end + 1].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor_cnhw[:, conv_output_shard_start : conv_output_shard_end + 1], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc

    return


def validate_utwh_output_shards_and_req_ds_input_shard_start_end(
    # input tensor shape
    input_nchw_shape,
    # DS golden output tensor to compare against
    out_golden_pyt_tensor,
    # Input indices corresponding to top left position of sliding window. Used to perform conv operation.
    data_top_left_indices,
    # validate utwh output shards
    utwh_output_shards,
    # Validate this config -
    req_ds_input_shard_start_end,
):
    output_n = out_golden_pyt_tensor.size()[0]
    output_c = out_golden_pyt_tensor.size()[1]
    output_h = out_golden_pyt_tensor.size()[2]
    output_w = out_golden_pyt_tensor.size()[3]
    assert len(data_top_left_indices) == output_n * output_h * output_w
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    assert output_n == input_n

    # Perform downsample on input shards one at a time, and compare against output. Use data_top_left_indices (global) to perform the downsample operation.
    output_stick_global = 0
    for input_shard_idx, item in enumerate(req_ds_input_shard_start_end):
        assert input_shard_idx < len(utwh_output_shards)
        ds_output_shard_start, ds_output_shard_end = item[0]
        req_ds_input_shard_start, req_ds_input_shard_end = item[1]
        # sanity check that the first item in the shard is at the top left position of sliding window
        assert output_stick_global < len(data_top_left_indices)
        assert req_ds_input_shard_start == data_top_left_indices[output_stick_global]
        output_shard = []
        output_shard_size = ds_output_shard_end - ds_output_shard_start + 1
        output_stick = output_stick_global
        for o in range(output_shard_size):
            assert output_stick < len(data_top_left_indices)
            input_top_left_position_stick = data_top_left_indices[output_stick]
            assert input_top_left_position_stick >= req_ds_input_shard_start
            input_shard_stick_local_idx = input_top_left_position_stick - req_ds_input_shard_start
            conv_input_window = []
            assert input_shard_stick_local_idx < len(utwh_output_shards[input_shard_idx])
            conv_input_window.append(utwh_output_shards[input_shard_idx][input_shard_stick_local_idx, :])
            output_shard.extend(utwh_output_shards[input_shard_idx][input_shard_stick_local_idx, :])
            output_stick += 1
        output_stick_global = output_stick
        output_pyt_shard = torch.tensor(output_shard).reshape((output_shard_size, input_c))
        # compare output shard with golden output pytorch tensor
        # permute output golden pytorch tensor from nchw to nhwc shape
        out_golden_pyt_tensor_nhwc = torch.permute(out_golden_pyt_tensor, (0, 2, 3, 1))
        # reshape cnhw to 2d shape = [nhw, c]
        out_golden_pyt_tensor_nhwc = torch.reshape(
            out_golden_pyt_tensor_nhwc, (output_n * output_h * output_w, output_c)
        )
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor_nhwc[ds_output_shard_start : ds_output_shard_end + 1, :].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor_nhwc[ds_output_shard_start : ds_output_shard_end + 1, :], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc

    return


def validate_tensor_metadata(
    input_tensor,
    input_nchw_shape,
    input_shard_size,
    tensor_metadata,
    req_conv_input_shard_start_end,
    golden_conv_input_shards,
):
    # input tensor is unpadded
    # Permute input tensor from nchw shape to nhwc shape and reshape to 2d shape - [nhw, c]
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    input_nhw_size = input_n * input_h * input_w
    input_tensor = np.reshape(input_tensor, input_nchw_shape)
    input_tensor_nhwc = np.transpose(input_tensor, (0, 2, 3, 1))
    input_tensor_nhwc = np.reshape(input_tensor_nhwc, (input_n * input_h * input_w, input_c))
    # construct unpadded input tensor shards
    unpadded_input_tensor_shards = []
    num_shards = len(req_conv_input_shard_start_end)
    unpadded_input_tensor_shard_start = 0
    for i in range(num_shards):
        unpadded_input_tensor_shard_end = min(unpadded_input_tensor_shard_start + input_shard_size, input_nhw_size)
        assert unpadded_input_tensor_shard_start < len(input_tensor_nhwc) and unpadded_input_tensor_shard_end <= len(
            input_tensor_nhwc
        )
        unpadded_input_tensor_shards.append(
            input_tensor_nhwc[unpadded_input_tensor_shard_start:unpadded_input_tensor_shard_end, :]
        )
        unpadded_input_tensor_shard_start += input_shard_size
    # Validate tensor_metadata
    # Construct conv input shard using tensor_metadata and req_conv_input_shard_start_end indices. Then, compare against golden conv input shards
    conv_input_shards = []
    assert len(req_conv_input_shard_start_end) == len(golden_conv_input_shards)
    for shard_idx, item in enumerate(req_conv_input_shard_start_end):
        conv_input_shard = []
        req_conv_input_shard_start = item[1][0]
        req_conv_input_shard_end = item[1][1]
        for idx in range(req_conv_input_shard_start, req_conv_input_shard_end + 1):
            assert idx < len(tensor_metadata)
            pad = tensor_metadata[idx][0]
            if pad:
                conv_input_shard.append([0] * input_c)
            else:
                core_id = tensor_metadata[idx][1]
                core_local_idx = tensor_metadata[idx][2]
                assert core_id < len(unpadded_input_tensor_shards)
                assert core_local_idx < len(unpadded_input_tensor_shards[core_id])
                conv_input_shard.append(unpadded_input_tensor_shards[core_id][core_local_idx, :])
        assert (conv_input_shard == golden_conv_input_shards[shard_idx]).all()
    return unpadded_input_tensor_shards


NEIGHBORHOOD_DIST = 2  ## ll, l, r, rr


def validate_untilize_with_halo_kernel_configs(
    golden,
    input_tensor_shards,
    resharded_start_and_end,
    local_data_start_and_size,
    local_pad_start_and_size,
    ll_send_start_and_size,
    l_send_start_and_size,
    r_send_start_and_size,
    rr_send_start_and_size,
    src_local_start_idx,
    local_data_nsegments_per_core,
    local_pad_nsegments_per_core,
    ll_data_nsegments_per_core,
    l_data_nsegments_per_core,
    r_data_nsegments_per_core,
    rr_data_nsegments_per_core,
    max_out_nsticks_per_core,
):
    ## using the kernel configs, construct the resulting resharding for each core
    ncores = len(resharded_start_and_end)
    assert len(input_tensor_shards) == ncores
    assert len(golden) == ncores
    input_c = len(golden[0][0])
    max_size = 0
    for _, dst in resharded_start_and_end:
        start = dst[0]
        end = dst[1]
        size = end - start + 1
        max_size = size if max_size < size else max_size
    pad_val = 0

    reshards = {}
    for core in np.arange(ncores):
        dst_range = resharded_start_and_end[core][1]
        curr_size = dst_range[1] - dst_range[0] + 1
        reshards[core] = np.zeros([curr_size, input_c])

    # print (f'RESHARD: {resharded_start_and_end}')
    for core in np.arange(ncores):
        local_data = local_data_start_and_size[core]
        local_pad = local_pad_start_and_size[core]
        ll_data = ll_send_start_and_size[core]
        l_data = l_send_start_and_size[core]
        r_data = r_send_start_and_size[core]
        rr_data = rr_send_start_and_size[core]
        src_start_idx = src_local_start_idx[core]
        ## local pad
        for local_pad_segment_idx in range(0, local_pad_nsegments_per_core[core] * 2, 2):
            dst_start = local_pad[local_pad_segment_idx]
            size = local_pad[local_pad_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core][dst_idx] = [pad_val] * input_c
                dst_idx += 1

        ## local data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST]
        for local_data_segment_idx in range(0, local_data_nsegments_per_core[core] * 2, 2):
            dst_start = local_data[local_data_segment_idx]
            size = local_data[local_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core][dst_idx] = input_tensor_shards[core][src_idx, :]  ## TODO: make global
                src_idx += 1
                dst_idx += 1

        ## push ll_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST - 2]
        for ll_data_segment_idx in range(0, ll_data_nsegments_per_core[core] * 2, 2):
            dst_start = ll_data[ll_data_segment_idx]
            size = ll_data[ll_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core - 2][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push l_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST - 1]
        for l_data_segment_idx in range(0, l_data_nsegments_per_core[core] * 2, 2):
            dst_start = l_data[l_data_segment_idx]
            size = l_data[l_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core - 1][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push r_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST + 1]
        for r_data_segment_idx in range(0, r_data_nsegments_per_core[core] * 2, 2):
            dst_start = r_data[r_data_segment_idx]
            size = r_data[r_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core + 1][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push rr_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST + 2]
        for rr_data_segment_idx in range(0, rr_data_nsegments_per_core[core] * 2, 2):
            dst_start = rr_data[rr_data_segment_idx]
            size = rr_data[rr_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core + 2][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

    assert max_out_nsticks_per_core == max([len(golden[core]) for core in range(ncores)])
    for core in np.arange(ncores):
        # print(f'OUTPUT CORE {core}: {reshards[core]}')
        # print(f'GOLDEN CORE {core}: {golden[core]}')
        assert (reshards[core] == golden[core]).all()
