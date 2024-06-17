// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "debug/dprint.h"
#include "debug/dprint_tensix.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK(( DPRINT << "======" << ENDL() ));
    for (uint16_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK(( DPRINT << (uint)r <<  " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
    }
    PACK(( DPRINT << "++++++" << ENDL() ));
}

inline void print_cb_details(uint32_t cb_id) {
    PACK(DPRINT << "cb_id " << cb_id << ": { "
            // << "size: " << cb_interface[cb_id].fifo_size << ", "
            // << "limit: " << cb_interface[cb_id].fifo_limit << ", "
            // << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
            // << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
            << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
            << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
            << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL() );
}

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CB::c_in0);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        // for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, per_core_block_dim);
            // print_full_tile(tt::CB::c_in0, 0);
            cb_pop_front(tt::CB::c_in0, per_core_block_dim);


            for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
                copy_tile(tt::CB::c_in0, tile_index, tile_index);
            }

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif
            // floor_tile_init();
            // for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            //     floor_tile(tile_index);
            // }
            // dprint_tensix_dest_reg(0);
            tile_regs_commit();

            tile_regs_wait();
            TTI_STALLWAIT(p_stall::STALL_PACK|p_stall::STALL_TDMA, p_stall::MATH);

            for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
                pack_tile(tile_index, tt::CB::c_out0);
            }
            print_full_tile(tt::CB::c_out0, 0);
            // print_cb_details(tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, per_core_block_dim);
            // TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);



            // release_dst(tt::DstMode::Half);
            tile_regs_release();
        // }
        // cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }

}
}
