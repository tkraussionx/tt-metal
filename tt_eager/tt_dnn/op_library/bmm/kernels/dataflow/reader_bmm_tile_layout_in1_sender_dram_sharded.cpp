// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

#include "debug/dprint.h"


template <uint32_t page_size, bool use_vc>
FORCE_INLINE
uint32_t noc_async_read_tile_dram_sharded_set_state(uint32_t bank_base_address, uint32_t bank_id = 0, const uint32_t vc = 0) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = bank_base_address + bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];

    DEBUG_STATUS("NRTW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(get_noc_addr_helper(src_noc_xy, src_addr_), dest_addr, page_size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS("NRTD");

    if constexpr(use_vc) {
        uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);  // len_bytes

    return src_addr_;
}

FORCE_INLINE
void noc_async_read_tile_dram_sharded_with_state(uint32_t src_base_addr, uint32_t src_addr, uint32_t dest_addr) {
    uint32_t src_addr_;

    src_addr_ = src_base_addr + src_addr;

    DEBUG_STATUS("NRTW");
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS("NRTD");

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr_);      // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;
}

template <uint32_t page_size, bool use_vc>
FORCE_INLINE
void noc_async_read_tile_dram_sharded(uint32_t bank_base_address, uint32_t src_addr, uint32_t dest_addr, uint32_t bank_id = 0, const uint32_t vc = 0) {
    uint32_t src_addr_;
    uint32_t src_noc_xy;

    src_addr_ = src_addr + bank_base_address;
    src_addr_ += bank_to_dram_offset[bank_id];
    src_noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];

    DEBUG_STATUS("NRTW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(get_noc_addr_helper(src_noc_xy, src_addr_), dest_addr, page_size);
    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));
    DEBUG_STATUS("NRTD");

    if constexpr(use_vc) {
        uint32_t noc_rd_cmd_field = NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CTRL, noc_rd_cmd_field);
    }

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dest_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, src_addr_);      // (uint32_t)src_addr
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, src_noc_xy);   // src_addr >> 32
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);  // len_bytes
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc_index] += 1;
}


void kernel_main() {
    // RUNTIME ARGS
    const uint32_t in1_tensor_addr                              = get_arg_val<uint32_t>(0);
    #ifdef FUSE_BIAS
    const uint32_t in3_tensor_addr                              = get_arg_val<uint32_t>(1);
    #endif
    const uint32_t dram_bank_id                                 = get_arg_val<uint32_t>(2);
    const uint32_t vc                                           = get_arg_val<uint32_t>(3);
    const uint32_t num_shard_to_write_back                        = get_arg_val<uint32_t>(4);
    const uint32_t reshard_tensor_start_offset                      = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t * per_core_N_reshard_bytes          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(6));
    volatile tt_l1_ptr uint32_t * in0_mcast_sender_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(7));
    volatile tt_l1_ptr uint32_t * in0_mcast_sender_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(8));


    // COMPILE TIME ARGS
    constexpr uint32_t in1_page_size                      = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_pages                      = get_compile_time_arg_val(1);
    // in1 block args
    constexpr uint32_t in1_block_w                        = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_num_tiles                = get_compile_time_arg_val(3);
    // in0/in1 common args
    constexpr uint32_t num_blocks                         = get_compile_time_arg_val(4);
    // WRITER
    constexpr uint32_t out_block_num_tiles                = get_compile_time_arg_val(5);
    constexpr uint32_t out_tensor_stride_w_bytes          = get_compile_time_arg_val(6);
    constexpr uint32_t out_reshard_tensor_stride_w_bytes  = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M                         = get_compile_time_arg_val(8);

    #ifdef FUSE_BIAS
    constexpr uint32_t in3_page_size                      = get_compile_time_arg_val(9);
    constexpr uint32_t in3_num_pages                      = get_compile_time_arg_val(10);
    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t bias_single_tile_size_bytes = get_tile_size(cb_id_in3);
    constexpr DataFormat bias_data_format = get_dataformat(cb_id_in3);
    #endif

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t cb_id_out_reshard = 17;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);

    //  READER
    uint32_t l1_write_addr_in1;
    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);

    // DPRINT << in1_page_size << ENDL();
    // DPRINT << in1_num_pages << ENDL();
    // DPRINT << in1_block_w << ENDL();
    // DPRINT << in1_block_num_tiles << ENDL();
    // DPRINT << in1_tensor_addr << ENDL();
    // DPRINT << in1_page_size << ENDL();
    // DPRINT << vc << ENDL();

    uint32_t in1_base_addr = noc_async_read_tile_dram_sharded_set_state<in1_page_size, true>(in1_tensor_addr, dram_bank_id, vc);

    uint32_t l1_read_addr_in1 = 0;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // DPRINT << l1_write_addr_in1 << ENDL();

        // Copy in1 block into CB, as the default kernel
        for(uint32_t h = 0; h < in1_num_pages; ++h) {
            // noc_async_read_tile_dram_sharded<in1_page_size, true>(in1_tensor_addr, l1_read_addr_in1, l1_write_addr_in1, dram_bank_id, vc);
            noc_async_read_tile_dram_sharded_with_state(in1_base_addr, l1_read_addr_in1, l1_write_addr_in1);
            l1_read_addr_in1 += in1_page_size;
            l1_write_addr_in1 += in1_page_size;
        }

        // Barrier! make sure the reads are done
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
    #ifdef FUSE_BIAS
        // Operand 1
        cb_reserve_back(cb_id_in3, in1_block_w);
        uint32_t l1_write_addr_in3 = get_write_ptr(cb_id_in3);
        uint32_t l1_read_addr_in3 = 0;

        uint32_t in3_base_addr = noc_async_read_tile_dram_sharded_set_state<in3_page_size, true>(in3_tensor_addr, dram_bank_id, vc);

        for(uint32_t h = 0; h < in3_num_pages; ++h) {
            // noc_async_read_tile_dram_sharded<in3_page_size, true>(in3_tensor_addr, l1_read_addr_in3, l1_write_addr_in3, dram_bank_id, vc);
            noc_async_read_tile_dram_sharded_with_state(in3_base_addr, l1_read_addr_in3, l1_write_addr_in3);
            l1_read_addr_in3 += in3_page_size;
            l1_write_addr_in3 += in3_page_size;
        }

        // Barrier! make sure the reads are done
        noc_async_read_barrier();
        cb_push_back(cb_id_in3, in1_block_w);
    #endif


    // WRITER
    cb_wait_front(cb_id_out, out_block_num_tiles);
    uint32_t index_offset = 0;
    uint32_t l1_read_addr_out_offset = 0;

    for (uint32_t i = 0; i < num_shard_to_write_back; ++i) {
        uint32_t l1_read_addr_out = get_read_ptr(cb_id_out) + l1_read_addr_out_offset;
        uint32_t l1_write_addr_out_reshard = get_write_ptr(cb_id_out_reshard);

        if (i == 0) {
            l1_write_addr_out_reshard += reshard_tensor_start_offset;
        }

        uint64_t reshard_dest_addr = get_noc_addr(in0_mcast_sender_noc_x[index_offset], in0_mcast_sender_noc_y[index_offset], l1_write_addr_out_reshard);

        for (uint32_t h = 0; h < per_core_M; ++h) {
            noc_async_write(l1_read_addr_out, reshard_dest_addr, per_core_N_reshard_bytes[index_offset]);
            l1_read_addr_out += out_tensor_stride_w_bytes;
            reshard_dest_addr += out_reshard_tensor_stride_w_bytes;
        }
        l1_read_addr_out_offset += per_core_N_reshard_bytes[index_offset];

        index_offset += 3;
    }
    noc_async_write_barrier();

}
