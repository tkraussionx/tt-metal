// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/experimental/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {

// inline void print_cb_details(uint32_t cb_id) {
//         UNPACK(DPRINT << "UNPACK cb_id " << cb_id << ": { "
//                 << "size: " << cb_interface[cb_id].fifo_size << ", "
//                 << "limit: " << cb_interface[cb_id].fifo_limit << ", "
//                 << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
//                 << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
//                 << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
//                 << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
//                 << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL());
// }

void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_in0, cb_intermed0, cb_out0);
    cb_wait_front(cb_intermed0, onetile);

    // num_output_tiles is 1
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        for (uint32_t j = 0; j < 8; j++) {
            PACK(DPRINT << "loop " << j << " / 8 " << ENDL());
            tile_regs_acquire();
            {
                cb_wait_front(cb_intermed0, onetile);
                copy_tile_to_dst_init_short(cb_intermed0);
                copy_tile(cb_intermed0, 0, 0);
                cb_pop_front(cb_intermed0, onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            {
                PACK(DPRINT << "before cb_reserve_back" << ENDL());
                cb_reserve_back(cb_intermed0, onetile); // Hang
                PACK(DPRINT << "after cb_reserve_back" << ENDL());
                pack_tile(0, cb_intermed0);
                cb_push_back(cb_intermed0, onetile);
            }
            tile_regs_release();
        }

        PACK(DPRINT << "pass the loop in compute kernel" << ENDL());

        // copy input tile from cb_in0 to cb_out0
        tile_regs_acquire();
        {
            cb_wait_front(cb_in0, onetile);
            unpack_reconfig_data_format_srca(cb_in0);
            copy_tile_to_dst_init_short(cb_in0);
            copy_tile(cb_in0, 0, 0);
            cb_pop_front(cb_in0, onetile);
            // print_cb_details(cb_in0);
        }
        tile_regs_commit();

        tile_regs_wait();
        {
            cb_reserve_back(cb_out0, onetile);
            pack_tile(0, cb_out0);
            cb_push_back(cb_out0, onetile);
        }
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
