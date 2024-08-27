// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    bool transpose_b = (get_arg_val<uint32_t>(7) == 1);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(8);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(9);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    // TODO: assign values to the variables
    uint32_t itileA = 0;
    uint32_t itileB = 0;
    uint32_t itileB_batch = output_tile_start_id % Nt;

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format};

    for (uint32_t n = 0; n < num_output_tiles; n++) {
        uint32_t itileC = output_tile_start_id + n;
        uint32_t mt = itileC / Nt;
        uint32_t nt = itileC % Nt;
        for (uint32_t kt = 0; kt < Kt; kt++) {
            itileA = mt * Kt + kt;
            itileB = kt * Nt + nt;
            // TODO:
            {  // Read A's tile at (mt, kt)
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(itileA, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }

            // TODO:
            {  // Read B's tile at (kt, nt)
                cb_reserve_back(cb_id_in1, onetile);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(itileB, s1, l1_write_addr_in1);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, onetile);
            }

            // TODO: Implement the indexing logic for itileA and itileB
            /*

            */
        }  // Kt loop

        itileB_batch += 1;
        // TODO: Implement the indexing logic for itileA and itileB
        /*

        */
    }
}
