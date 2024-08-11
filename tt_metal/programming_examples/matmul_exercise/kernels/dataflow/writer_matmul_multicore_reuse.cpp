// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {


    // out tensor args
    uint32_t out_tensor_addr                         = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id                = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w                     = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h                     = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w       = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h       = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w                   = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h                   = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count          = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w              = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h              = get_arg_val<uint32_t>(10);

    constexpr bool out_is_dram = get_compile_time_arg_val(0)== 1;

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);
    // TODO: Fill the /* */ with appropriate code
    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = /* */,
        .page_size = /* */,
        .data_format = /* */
    };


    // TODO: Fill the /* */ with appropriate code
    uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
    for(uint32_t sbh = 0; sbh < /* */; sbh++) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for(uint32_t sbw = 0; sbw < /* */; sbw++) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

            cb_wait_front(cb_id_out0, out_subblock_tile_count);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

            for(uint32_t h = 0; h < /* */; h++) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for(uint32_t w = 0; w < /* */; w++) {
                    noc_async_write_tile(/* */, /* */, /* */);
                    l1_read_addr+=/* */;

                    out_tensor_tile_id += /* */;
                }
                out_tensor_sb_row_start_tile_id += /* */;
            }

            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, out_subblock_tile_count);
            out_tensor_sbw_start_tile_id += /* */;
        }
        out_tensor_sbh_start_tile_id += /* */;
    }
}
