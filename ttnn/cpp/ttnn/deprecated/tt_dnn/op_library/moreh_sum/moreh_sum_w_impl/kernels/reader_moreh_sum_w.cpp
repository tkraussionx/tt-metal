// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

FORCE_INLINE void generate_reduce_scaler_fp32(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 4096 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    // TODO: src addr does not need to be rewritten. Update/add api for this
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    if (scaler != 0) {
        for (int k = 0; k < 4; ++k) {
            uint32_t idx = k << 7;
            for (int j = 0; j < 16; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }

    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t mask_w = get_arg_val<uint32_t>(3);
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t scaler = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_in1 = 1;  // cb_zeros
    constexpr uint32_t cb_id_in2 = 2;  // cb_scaler

    generate_reduce_scaler(cb_id_in1, 0);
#ifdef FP32_DEST_ACC_EN
    generate_reduce_scaler_fp32(cb_id_in2, scaler);
#else
    generate_reduce_scaler(cb_id_in2, scaler);
#endif

    constexpr uint32_t cb_id_mask_w = 3;
#ifdef DO_MASK_W
    generate_mask_w(cb_id_mask_w, mask_w);
#endif

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
