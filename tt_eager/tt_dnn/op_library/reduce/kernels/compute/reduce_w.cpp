// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"


inline void print_cb_details(uint32_t cb_id) {
    DPRINT << "cb_id " << cb_id << ": { "
            << "size: " << cb_interface[cb_id].fifo_size << ", "
            << "limit: " << cb_interface[cb_id].fifo_limit << ", "
            << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
            << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
            << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
            << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
            << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
}

namespace NAMESPACE {

union floatconverter {
    float f;
    uint32_t i;
};

void slow_reduce_sum_w(uint32_t cb_inp, uint32_t cb_out, uint32_t num_inps) {
    volatile tt_l1_ptr uint16_t* inp_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_interface[cb_inp].fifo_rd_ptr << 4);
    volatile tt_l1_ptr uint16_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_interface[cb_out].fifo_rd_ptr << 4);
    float sums[32] = {0};
    constexpr uint32_t face_offset = 16 * 16;
    for (uint32_t tile = 0; tile < num_inps; ++tile) {
        for (uint32_t r = 0; r < 32; ++r) {
            float sum = sums[r];
            uint32_t row_face_offset = r < 16 ? 0 : 2 * face_offset;
            uint32_t row_offset = row_face_offset + (r % 16) * 16;
            for (uint32_t c = 0; c < 32; ++c) {
                uint32_t col_face_offset = c < 16 ? 0 : face_offset;
                uint32_t col_offset = col_face_offset + c % 16;
                uint16_t val = inp_ptr[tile * 32 * 32 + row_offset + col_offset];
                floatconverter fc;
                fc.i = val << 16;
                float fval = fc.f;
                // DPRINT << "tile = " << tile << " r = " << r << " c = " << c << " row_offset = " << row_offset << " col_offset = " << col_offset << ENDL();
                // DPRINT << "fval = " << fval << " val = " << val << " sum = " << sum << ENDL();
                sum += fval;
            }
            sums[r] = sum;
        }
    }

    for (uint32_t r = 0; r < 32; ++r) {
        // DPRINT << "sums[" << r << "] = " << sums[r] << ENDL();
        uint32_t row_face_offset = r < 16 ? 0 : 2 * face_offset;
        uint32_t row_offset = row_face_offset + (r % 16) * 16;
        floatconverter fc;
        fc.f = sums[r];
        // DPRINT << "fc.f = " << fc.f << ENDL();
        // DPRINT << "out_ptr: " << static_cast<uint32_t>(reinterpret_cast<uintptr_t>(out_ptr)) + row_offset << ENDL();
        out_ptr[row_offset] = (uint16_t)(fc.i >> 16);
    }

}

void MAIN {

    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    reduce_init<true>(REDUCE_OP, REDUCE_DIM, tt::CB::c_in0, tt::CB::c_in2);


    cb_wait_front(tt::CB::c_in2, 1); // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            cb_reserve_back(tt::CB::c_out0, onetile);
            cb_wait_front(tt::CB::c_in0, Wt);

            // UNPACK(DPRINT << "c_in0" << ENDL());
            // UNPACK(print_cb_details(tt::CB::c_in0));
            // UNPACK(DPRINT << "c_out0" << ENDL());
            // UNPACK(print_cb_details(tt::CB::c_out0));
            acquire_dst(tt::DstMode::Half);
            UNPACK(slow_reduce_sum_w(tt::CB::c_in0, tt::CB::c_out0, Wt));
            tensix_sync();
            reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, 0); // Force Math to wait for unpacker to finish
            release_dst(tt::DstMode::Half);
            tensix_sync();
            PACK(DPRINT << "Pack dprint!" << ENDL());
            cb_pop_front(tt::CB::c_in0, Wt);

            cb_push_back(tt::CB::c_out0, onetile);
        }
    }
}
}
