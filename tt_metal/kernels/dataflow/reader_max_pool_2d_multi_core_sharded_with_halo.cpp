// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"

#include "debug_print.h"

// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1 };

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     DPRINT << "======" << ENDL();
//     for (int32_t r = 0; r < 32; ++ r) {
//         SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 64, .ws = 2};
//         DPRINT << (uint) r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
//     }
//     DPRINT << "++++++" << ENDL();
// }

// inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages) {
//     volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
//     for (uint32_t page = 0; page < npages; ++ page) {
//         for (uint32_t i = 0; i < pagelen; ++ i) {
//             if (i == 0) {
//                 DPRINT << BF16(*ptr) << " ";
//             }
//             ptr ++;
//         }
//         DPRINT << ENDL();
//     }
// }

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

inline bool fill_with_val_async(const InterleavedPow2AddrGenFast<false>& s_const, uint32_t begin_addr, int32_t nrows, uint32_t row_nbytes) {
    uint32_t curr_addr = begin_addr;
    for (int32_t row_i = 0; row_i < nrows; ++ row_i) {
        s_const.noc_async_read_page(0, curr_addr);
        curr_addr += row_nbytes;
    }
    return true;
}

/**
 * Max-pool 2D.
 */
void kernel_main() {
    // input tensor address
    const uint32_t in_addr = get_arg_val<uint32_t>(0);

    const uint32_t window_h = get_arg_val<uint32_t>(2);
    const uint32_t window_w = get_arg_val<uint32_t>(3);
    const int32_t window_hw = get_arg_val<int32_t>(4);
    // window_hw_padded = window_hw rounded up to the tile size (can be multiple tiles)
    const uint32_t window_hw_padded = get_arg_val<uint32_t>(5);

    const int32_t pad_h = get_arg_val<int32_t>(8);
    const int32_t pad_w = get_arg_val<int32_t>(9);

    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);

    // channel size in bytes, multiple of 32
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(14);

    // input tensor height / width / channels
    const int32_t in_h = get_arg_val<int32_t>(16);
    const int32_t in_w = get_arg_val<int32_t>(17);
    const int32_t in_c = get_arg_val<int32_t>(19);

    const int32_t in_cb_pagesize = get_arg_val<int32_t>(22);
    // product of window_hw_padded and in_c padded to the tile size (can be multiple tiles)
    const int32_t in_cb_page_nelems_padded = get_arg_val<int32_t>(24);

    // out_w divided by number of out_nelems (== number of blocks per iteration)
    const int32_t out_w_loop_count = get_arg_val<int32_t>(25);
    const uint32_t in_log_base_2_of_page_size = get_arg_val<uint32_t>(26);

    const uint32_t nbatch = get_arg_val<uint32_t>(27);

    const uint32_t in_hw = get_arg_val<uint32_t>(28);

    const uint32_t minus_inf_buffer_addr = get_arg_val<uint32_t>(34);
    const uint32_t minus_inf_buffer_nbytes = get_arg_val<uint32_t>(35);
    const uint32_t in_cb_nsticks = get_arg_val<uint32_t>(36);

    // the starting offset for assigned batch input row id (batch_offset)
    uint32_t core_offset_in_stick_id = get_arg_val<uint32_t>(37);

    // compile time args
    constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;
    // value of 1 in bf16 in a uin32_t
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(2);
    // number of output elements per iteration == number of blocks per iteration
    constexpr uint32_t out_nelems = get_compile_time_arg_val(3);
    constexpr bool use_pow2 = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    constexpr uint32_t reader_noc = get_compile_time_arg_val(7);
    constexpr uint32_t writer_noc = get_compile_time_arg_val(8);

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;
    constexpr uint32_t in_shard_cb_id = tt::CB::c_in2;    // local input shard

    constexpr uint32_t TILE_HW = 1024;

    DPRINT << "HAHA 0" << ENDL();

    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    // fill 1 tile w/ scalar
    fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_HW, bf16_one_u16);
    cb_push_back(in_scalar_cb_id, 1);

    // fill in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    const InterleavedPow2AddrGenFast<false> s_const = {     // NOTE: This is always in L1 (hardcoded in host)
        .bank_base_address = minus_inf_buffer_addr,
        .log_base_2_of_page_size = in_log_base_2_of_page_size        // TODO: generalize?, currently hardcorded for 1 row of 32 16b values
    };
    fill_with_val_async(s_const, in_l1_write_addr, in_cb_nsticks, in_nbytes_c);
    noc_async_read_barrier();

    // NOTE: batch is folded in

    DPRINT << "HAHA 1" << ENDL();

    uint32_t core_out_w_i_start = get_arg_val<int32_t>(38);
    uint32_t core_out_h_i_start = get_arg_val<int32_t>(39);
    uint32_t nsticks_per_core = get_arg_val<uint32_t>(40);

    uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(42);

    uint32_t local_out_stick_start = get_arg_val<uint32_t>(43);
    uint32_t nsticks_per_batch = get_arg_val<uint32_t>(44);
    uint32_t local_in_stick_start = get_arg_val<uint32_t>(45);
    uint32_t local_in_stick_end = get_arg_val<uint32_t>(46);
    uint32_t in_nsticks_per_batch = get_arg_val<uint32_t>(47);
    uint32_t in_nsticks_per_core = get_arg_val<uint32_t>(48);

    uint32_t has_left = get_arg_val<uint32_t>(49);
    uint32_t left_noc_x = get_arg_val<uint32_t>(50);
    uint32_t left_noc_y = get_arg_val<uint32_t>(51);
    uint32_t has_right = get_arg_val<uint32_t>(52);
    uint32_t right_noc_x = get_arg_val<uint32_t>(53);
    uint32_t right_noc_y = get_arg_val<uint32_t>(54);

    // TODO: pass these as runtime args
    uint32_t in_nbytes_c_log2 = 7;  // for in_nbytes_c == 128
    // for in_nsticks_per_core == 1024, remainder mask = 0x3ff
    // uint32_t in_nsticks_per_core_rem_mask = 0x3ff;
    uint32_t in_nsticks_per_core_rem_mask = get_arg_val<uint32_t>(55);

    uint32_t has_left_left = get_arg_val<uint32_t>(56);
    uint32_t left_left_noc_x = get_arg_val<uint32_t>(57);
    uint32_t left_left_noc_y = get_arg_val<uint32_t>(58);
    uint32_t has_right_right = get_arg_val<uint32_t>(59);
    uint32_t right_right_noc_x = get_arg_val<uint32_t>(60);
    uint32_t right_right_noc_y = get_arg_val<uint32_t>(61);
    uint32_t left_in_stick_start = get_arg_val<uint32_t>(62);
    uint32_t right_in_stick_end = get_arg_val<uint32_t>(63);

    int32_t my_core = get_arg_val<int32_t>(64);

    uint32_t partial_first_row_nsticks = get_arg_val<uint32_t>(65);
    uint32_t partial_first_row_skip = get_arg_val<uint32_t>(66);
    uint32_t partial_top_image_nrows = get_arg_val<uint32_t>(67);
    uint32_t partial_top_image_skip = get_arg_val<uint32_t>(68);
    uint32_t full_nimages = get_arg_val<uint32_t>(69);
    uint32_t full_images_skip = get_arg_val<uint32_t>(70);
    uint32_t partial_bottom_image_nrows = get_arg_val<uint32_t>(71);
    uint32_t partial_last_row_nsticks = get_arg_val<uint32_t>(72);

    constexpr uint32_t cb_reader_indices = tt::CB::c_intermed0;
    volatile tt_l1_ptr uint32_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t top_left_i = 0;
    uint32_t reader_i = 0;

    DPRINT << "HAHA 2" << ENDL();
    // section 1: partial first row
    for (uint32_t i = 0; i < partial_first_row_nsticks; ++ i) {
        reader_indices_ptr[reader_i ++] = top_left_i ++;
    }
    top_left_i += partial_first_row_skip;

    DPRINT << "HAHA 3" << ENDL();
    // section 2: partial first image
    for (uint32_t i = 0; i < partial_top_image_nrows; ++ i) {
        for (int32_t j = 0; j < out_w; ++ j) {
            reader_indices_ptr[reader_i ++] = top_left_i ++;
        }
        top_left_i += window_w - 1;
    }
    top_left_i += partial_top_image_skip;

    DPRINT << "HAHA 4" << ENDL();
    // section 3: full images
    for (uint32_t n = 0; n < full_nimages; ++ n) {
        for (int32_t i = 0; i < out_h; ++ i) {
            for (int32_t j = 0; j < out_w; ++ j) {
                reader_indices_ptr[reader_i ++] = top_left_i ++;
            }
            top_left_i += window_w - 1;
        }
        top_left_i += full_images_skip;
    }

    DPRINT << "HAHA 5" << ENDL();
    // section 4: partial last image
    for (uint32_t i = 0; i < partial_bottom_image_nrows; ++ i) {
        for (int32_t j = 0; j < out_w; ++ j) {
            reader_indices_ptr[reader_i ++] = top_left_i ++;
        }
        top_left_i += window_w - 1;
    }

    DPRINT << "HAHA 6" << ENDL();
    // section 5: partial last row
    for (uint32_t i = 0; i < partial_last_row_nsticks; ++ i) {
        reader_indices_ptr[reader_i ++] = top_left_i ++;
    }

    DPRINT << "reader_i = " << reader_i << ENDL();
    for (uint32_t i = 0; i < reader_i; ++ i) {
        DPRINT << reader_indices_ptr[i] << ENDL();
    }

} // kernel_main()
