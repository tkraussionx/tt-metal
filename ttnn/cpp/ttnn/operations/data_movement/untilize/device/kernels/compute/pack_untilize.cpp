// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
//    UNPACK(DPRINT << "IN PACK UNTILIZE " << ENDL());
    UNPACK(( DPRINT << "Block count=" << uint32_t(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt << ENDL() ));

    pack_untilize_init<per_core_block_tile_cnt>(tt::CB::c_in0, tt::CB::c_out0);


    for(uint32_t b = 0; b < per_core_block_cnt; ++ b) {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);
//        // Print a full tile
//        for(uint32_t i=0; i< per_core_block_tile_cnt; i++) {
//            for (uint16_t r = 0; r < 32; ++r) {
//                uint16_t r_next = r + 1;
//                SliceRange sr =  SliceRange{.h0 = r, .h1 = r_next, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
//                UNPACK(DPRINT << (uint)r << " --READ--cin0-- " << TileSlice(tt::CB::c_in0, i, sr, true, true) << ENDL());
//            }
//        }
        pack_untilize_block<per_core_block_tile_cnt>(tt::CB::c_in0, 0, tt::CB::c_out0);



        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_wait_front(tt::CB::c_out0, per_core_block_tile_cnt);

        // Print a full tile
        //for(uint32_t i=0; i< per_core_block_tile_cnt; i++) {
        //    for (uint16_t r = 0; r < 32; ++r) {
        //        uint16_t r_next = r + 1;
        //        SliceRange sr =  SliceRange{.h0 = r, .h1 = r_next, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //        UNPACK(DPRINT << (uint)r << " --READ--cout0-- " << TileSlice(tt::CB::c_out0, i, sr, true, false) << ENDL());
        //    }
        //}

    }

    pack_untilize_uninit();
}
}
