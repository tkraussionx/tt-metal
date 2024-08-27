// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(16);

    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // TODO: Fill the /* */ with appropriate code
    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr /* */,
        .page_size = in0_single_tile_size_bytes /* */,
        .data_format = in0_data_format /* */
    };

    // TODO: Fill the /* */ with appropriate code
    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = in1_tensor_addr /* */,
        .page_size = in1_single_tile_size_bytes /* */,
        .data_format = in0_data_format /* */
    };

    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);

        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // TODO: Fill the /* */ with appropriate code
        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in0_block_h; h++) {
            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
            for (uint32_t w = 0; w < in0_block_w; w++) {
                noc_async_read_tile(in0_tensor_tile_id /* */, s0 /* */, l1_write_addr_in0 /* */);
                // CHECK
                l1_write_addr_in0 += in0_single_tile_size_bytes /* */;
                in0_tensor_tile_id += 1 /* */;
            }
            in0_tensor_row_start_tile_id += in0_tensor_stride_h /* */;
        }
        in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride /* */;

        // TODO: Implement the same logic as above for in1, replacing in0_ with in1_ accordingly
        /*

        for (..) {
          for (...) {
            noc_async_read_tile(..., s1, ...);
          }
        }

        */
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for (uint32_t w = 0; w < in1_block_w; w++) {
                noc_async_read_tile(in1_tensor_tile_id /* */, s1 /* */, l1_write_addr_in1 /* */);
                // CHECK
                l1_write_addr_in1 += in1_single_tile_size_bytes /* */;
                in1_tensor_tile_id += 1 /* */;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h /* */;
        }
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride /* */;

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
