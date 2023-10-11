// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);  // has input shard
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1); // has output shard with padding and halo

    // for this core's local shard
    uint32_t in_nsticks = get_arg_val<uint32_t>(0);
    uint32_t out_nsticks = get_arg_val<uint32_t>(1);

    uint32_t partial_first_row_nsticks = get_arg_val<uint32_t>(2);
    uint32_t pad_w = get_arg_val<uint32_t>(3);
    uint32_t out_w = get_arg_val<uint32_t>(4);
    uint32_t partial_top_image_nrows = get_arg_val<uint32_t>(5);
    uint32_t pad_h = get_arg_val<uint32_t>(6);
    uint32_t out_h = get_arg_val<uint32_t>(7);
    uint32_t full_nimages = get_arg_val<uint32_t>(8);
    uint32_t partial_bottom_image_nrows = get_arg_val<uint32_t>(9);
    uint32_t partial_last_row_nsticks = get_arg_val<uint32_t>(10);
    uint32_t halo_for_left_left_nsticks = get_arg_val<uint32_t>(11);
    uint32_t halo_for_left_nsticks = get_arg_val<uint32_t>(12);
    uint32_t halo_for_right_nsticks = get_arg_val<uint32_t>(13);
    uint32_t halo_for_right_right_nsticks = get_arg_val<uint32_t>(14);

    uint32_t local_in_stick_start = get_arg_val<uint32_t>(15);
    uint32_t local_in_stick_end = get_arg_val<uint32_t>(16);
    uint32_t in_nsticks_per_batch = get_arg_val<uint32_t>(17);
    uint32_t in_nsticks_per_core = get_arg_val<uint32_t>(18);

    uint32_t has_left = get_arg_val<uint32_t>(19);
    uint32_t left_noc_x = get_arg_val<uint32_t>(20);
    uint32_t left_noc_y = get_arg_val<uint32_t>(21);
    uint32_t has_right = get_arg_val<uint32_t>(22);
    uint32_t right_noc_x = get_arg_val<uint32_t>(23);
    uint32_t right_noc_y = get_arg_val<uint32_t>(24);
    uint32_t has_left_left = get_arg_val<uint32_t>(25);
    uint32_t left_left_noc_x = get_arg_val<uint32_t>(26);
    uint32_t left_left_noc_y = get_arg_val<uint32_t>(27);
    uint32_t has_right_right = get_arg_val<uint32_t>(28);
    uint32_t right_right_noc_x = get_arg_val<uint32_t>(29);
    uint32_t right_right_noc_y = get_arg_val<uint32_t>(30);

    uint32_t stick_nbytes = get_arg_val<uint32_t>(31);   // size of 1 stick (in_c_nbytes)

    // nsticks to push to left left neighbor core
    uint32_t left_left_core_nsticks = get_arg_val<uint32_t>(32);
    // nsticks to push to left neighbor core
    uint32_t left_core_nsticks = get_arg_val<uint32_t>(33);
    // nsticks to push to right neighbor core
    uint32_t right_core_nsticks = get_arg_val<uint32_t>(34);
    // nsticks to push to right right neighbor core
    uint32_t right_right_core_nsticks = get_arg_val<uint32_t>(35);

    // offset on left left neighbor core for its right right halo
    uint32_t left_left_core_halo_offset = get_arg_val<uint32_t>(36);
    // offset on left neighbor core for its right halo
    uint32_t left_core_halo_offset = get_arg_val<uint32_t>(37);
    // offset on right neighbor core for its left halo
    uint32_t right_core_halo_offset = get_arg_val<uint32_t>(38);
    // offset on right right neighbor core for its left left halo
    uint32_t right_right_core_halo_offset = get_arg_val<uint32_t>(39);

    // padding (2) index offset in the halo going to left neighbors
    uint32_t left_going_halo_pad_i_offset = get_arg_val<uint32_t>(40);
    // padding (2) index offset in the halo going to right neighbors
    uint32_t right_going_halo_pad_i_offset = get_arg_val<uint32_t>(41);

    uint32_t pad_val_buffer_l1_addr = get_arg_val<uint32_t>(42);

    // 1. (partial first row width + pad_w)
    // 2. (out_w + pad_w * 2) * (num full rows partial top image)
    // 3. (out_w + pad_w * 2) * (pad_h + out_h) * num full images
    // 4. (out_w + pad_w * 2) * (pad_h + num full rows partial bottom image)
    // 5. (partial last row width + pad_w)

    // DPRINT << "0" << ENDL();

    cb_wait_front(in_cb_id, in_nsticks);

    uint32_t in_l1_addr = get_read_ptr(in_cb_id);
    uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);
    uint64_t padding_noc_addr = get_noc_addr(pad_val_buffer_l1_addr);

    uint32_t halo_nsticks = out_w + 2 * pad_w;

    // DPRINT << "1" << ENDL();

    // section 1
    uint32_t section1_l1_addr = out_base_l1_addr + (halo_nsticks + 1) * stick_nbytes;
    uint32_t curr_out_l1_addr = section1_l1_addr;
    uint32_t curr_in_l1_addr = in_l1_addr;
    for (uint32_t i = 0; i < partial_first_row_nsticks; ++ i) {
        uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
        noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes);
        curr_in_l1_addr += stick_nbytes;
        curr_out_l1_addr += stick_nbytes;
    }
    // insert padding stick at the end of the row
    noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);

    // DPRINT << "2" << ENDL();

    // section 2
    for (uint32_t i = 0; i < partial_top_image_nrows; ++ i) {
        // padding sticks on the left
        for (uint32_t j = 0; j < pad_w; ++ j) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
        // data sticks for full row
        for (uint32_t j = 0; j < out_w; ++ j) {
            uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
            noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_in_l1_addr += stick_nbytes;
            curr_out_l1_addr += stick_nbytes;
        }
        // padding sticks on the right
        for (uint32_t j = 0; j < pad_w; ++ j) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
    }

    // DPRINT << "3" << ENDL();

    // section 3
    for (uint32_t n = 0; n < full_nimages; ++ n) {
        // padding rows on top
        for (uint32_t i = 0; i < pad_h; ++ i) {
            // padding sticks on the left
            for (uint32_t j = 0; j < pad_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
            // padding full row
            for (uint32_t j = 0; j < out_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
            // padding sticks on the right
            for (uint32_t j = 0; j < pad_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
        }
        // full image rows
        for (uint32_t i = 0; i < out_h; ++ i) {
            // padding sticks on the left
            for (uint32_t j = 0; j < pad_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
            // data sticks for full row
            for (uint32_t j = 0; j < out_w; ++ j) {
                uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
                noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_in_l1_addr += stick_nbytes;
                curr_out_l1_addr += stick_nbytes;
            }
            // padding sticks on the right
            for (uint32_t j = 0; j < pad_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
        }
    }

    // DPRINT << "4" << ENDL();

    // section 4
    for (uint32_t i = 0; i < partial_bottom_image_nrows; ++ i) {
        // padding sticks on the left
        for (uint32_t j = 0; j < pad_w; ++ j) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
        // data sticks for full row
        for (uint32_t j = 0; j < out_w; ++ j) {
            uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
            noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_in_l1_addr += stick_nbytes;
            curr_out_l1_addr += stick_nbytes;
        }
        // padding sticks on the right
        for (uint32_t j = 0; j < pad_w; ++ j) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
    }

    // DPRINT << "5" << ENDL();

    // section 5
    // insert padding stick at the beginning of the row
    noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
    // partial row sticks
    for (uint32_t i = 0; i < partial_last_row_nsticks; ++ i) {
        uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
        noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes);
        curr_in_l1_addr += stick_nbytes;
        curr_out_l1_addr += stick_nbytes;
    }

    // Local sticks that are also part of halo for the left/right neighbors
    // NOTE: assuming the base l1 addr are the same on all cores

    // DPRINT << "6" << ENDL();

    // section B (push halo to right and right right neighbors cores)
    curr_in_l1_addr = curr_in_l1_addr - (out_w + 1) * stick_nbytes;  // rewind by -(out_w + 1)
    uint32_t right_i = 0;
    if (has_right) {
        DPRINT << "HALO TO R = " << right_core_nsticks << " (" << right_noc_x << "," << right_noc_y << "): ";
        uint32_t out_l1_addr_right = out_base_l1_addr + right_core_halo_offset;
        // push sticks to right neighbor
        for (uint32_t i = 0; i < right_core_nsticks; ++ i, ++ right_i) {
            if (right_i == right_going_halo_pad_i_offset) {
                // send padding sticks (2 * pad_w)
                uint64_t noc_addr = get_noc_addr(right_noc_x, right_noc_y, out_l1_addr_right);
                // TODO: may be the remote core can fill padding locally for its halo ...
                // TODO: fix padding source...
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right += stick_nbytes;
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right += stick_nbytes;
                // increament the nsticks to offset due to padding
                right_core_nsticks += 2 * pad_w;
                DPRINT << "P P ";
            } else {
                uint64_t noc_addr = get_noc_addr(right_noc_x, right_noc_y, out_l1_addr_right);
                noc_async_write(curr_in_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right += stick_nbytes;
                curr_in_l1_addr += stick_nbytes;
                DPRINT << right_i << " ";
            }
        }
        DPRINT << ENDL();
    }
    if (has_right_right) {
        DPRINT << "HALO TO RR = " << right_right_core_nsticks << " (" << right_right_noc_x << "," << right_right_noc_y << "): ";
        uint32_t out_l1_addr_right_right = out_base_l1_addr + right_right_core_halo_offset;
        // push sticks to right right neighbor
        for (uint32_t i = 0; i < right_right_core_nsticks; ++ i, ++ right_i) {
            if (right_i == right_going_halo_pad_i_offset) {
                // send padding sticks (2 * pad_w)
                uint64_t noc_addr = get_noc_addr(right_right_noc_x, right_right_noc_y, out_l1_addr_right_right);
                // TODO: may be the remote core can fill padding locally for its halo ...
                // TODO: fix padding source...
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right_right += stick_nbytes;
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right_right += stick_nbytes;
                // increament the nsticks to offset due to padding
                right_right_core_nsticks += 2 * pad_w;
                DPRINT << "P P ";
            } else {
                uint64_t noc_addr = get_noc_addr(right_right_noc_x, right_right_noc_y, out_l1_addr_right_right);
                noc_async_write(curr_in_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_right_right += stick_nbytes;
                curr_in_l1_addr += stick_nbytes;
                DPRINT << right_i << " ";
            }
        }
        DPRINT << ENDL();
    }

    // DPRINT << "7" << ENDL();

    // section A (push halo to left and left left neighbors)
    curr_in_l1_addr = in_l1_addr;   // reset to the base
    uint32_t left_i = 0;
    if (has_left_left) {
        DPRINT << "HALO TO LL = " << left_left_core_nsticks << " (" << left_left_noc_x << "," << left_left_noc_y << "): ";
        // these sticks belong to the right right halo of the left left neighbor
        uint32_t out_l1_addr_left_left = out_base_l1_addr + left_left_core_halo_offset;
        // push sticks to left left neighbor
        for (uint32_t i = 0; i < left_left_core_nsticks; ++ i, ++ left_i) {
            if (left_i == left_going_halo_pad_i_offset) {
                // send padding sticks (2 * pad_h)
                uint64_t noc_addr = get_noc_addr(left_left_noc_x, left_left_noc_y, out_l1_addr_left_left);
                // TODO: may be the remote core can fill padding locally for its halo ...
                // TODO: fix padding source...
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_left_left += stick_nbytes;
                // increament the nsticks to offset due to padding
                left_left_core_nsticks += 2 * pad_h;
                DPRINT << "P P ";
            } else {
                uint64_t noc_addr = get_noc_addr(left_left_noc_x, left_left_noc_y, out_l1_addr_left_left);
                noc_async_write(curr_in_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_left_left += stick_nbytes;
                curr_in_l1_addr += stick_nbytes;
                DPRINT << left_i << " ";
            }
        }
        DPRINT << ENDL();
    }
    if (has_left) {
        DPRINT << "HALO TO L = " << left_core_nsticks << " (" << left_noc_x << "," << left_noc_y << "): ";
        // these sticks belong to the right halo of the left neighbor
        uint32_t out_l1_addr_left = out_base_l1_addr + left_core_halo_offset;
        // send sticks to left left neighbor
        for (uint32_t i = 0; i < left_core_nsticks; ++ i, ++ left_i) {
            if (left_i == left_going_halo_pad_i_offset) {
                // send padding sticks (2 * pad_h)
                uint64_t noc_addr = get_noc_addr(left_noc_x, left_noc_y, out_l1_addr_left);
                // TODO: may be the remote core can fill padding locally for its halo ...
                // TODO: fix padding source...
                // noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_left += stick_nbytes;
                // increament the nsticks to offset due to padding
                left_core_nsticks += 2 * pad_h;
                DPRINT << "P P ";
            } else {
                uint64_t noc_addr = get_noc_addr(left_noc_x, left_noc_y, out_l1_addr_left);
                noc_async_write(curr_in_l1_addr, noc_addr, stick_nbytes);
                out_l1_addr_left += stick_nbytes;
                curr_in_l1_addr += stick_nbytes;
                DPRINT << left_i << " ";
            }
        }
        DPRINT << ENDL();
    }

    noc_async_read_barrier();
    noc_async_write_barrier();
}
