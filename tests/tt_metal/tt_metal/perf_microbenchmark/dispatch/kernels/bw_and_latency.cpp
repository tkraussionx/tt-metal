// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// void kernel_main() {
// #ifdef PAGE_SIZE
//     uint32_t page_size = PAGE_SIZE;
// #else
//     uint32_t page_size = get_arg_val<uint32_t>(0);
// #endif
//     cb_reserve_back(0, PAGE_COUNT);
//     uint32_t cb_addr = get_write_ptr(0);
//     for (int i = 0; i < ITERATIONS; i++) {
//         uint32_t read_ptr = cb_addr;
//         for (int j = 0; j < PAGE_COUNT; j++) {
// #if DRAM_BANKED
//             uint64_t noc_addr = get_dram_noc_addr(j, page_size, 0);
// #else
//             uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
// #endif
// #if READ_ONE_PACKET
//             noc_async_read_one_packet(noc_addr, read_ptr, page_size);
// #else
//             noc_async_read(noc_addr, read_ptr, page_size);
// #endif
// #if LATENCY
//             noc_async_read_barrier();
// #endif
//             read_ptr += page_size;
//         }
//     }
// #if !LATENCY
//     noc_async_read_barrier();
// #endif
// }

constexpr uint32_t scratch_db_size = SCRATCH_DB_SIZE;
constexpr uint32_t scratch_db_half_size = SCRATCH_DB_SIZE / 4;

FORCE_INLINE
void ncrisc_noc_fast_read_helper(uint32_t noc, uint32_t cmd_buf, uint64_t src_addr, uint32_t dest_addr, uint32_t len_bytes) {
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_addr);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)src_addr);
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT));
  NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
//   noc_reads_num_issued[noc] += 1;
}


FORCE_INLINE
void cq_noc_async_read_with_trid(std::uint64_t src_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint32_t trid) {
    while (!noc_cmd_buf_ready(1, NCRISC_RD_CMD_BUF));
    // ncrisc_noc_set_transaction_id(1, NCRISC_RD_CMD_BUF, trid);
    ncrisc_noc_fast_read_helper(1, NCRISC_RD_CMD_BUF, src_addr, dst_local_l1_addr, size);
}

FORCE_INLINE
void cq_noc_async_read_with_trid_any_len(std::uint64_t src_addr, std::uint32_t dst_local_l1_addr, std::uint32_t size, uint32_t trid) {
    while (size > NOC_MAX_BURST_SIZE) {
        cq_noc_async_read_with_trid(src_addr, dst_local_l1_addr, NOC_MAX_BURST_SIZE, trid);
        src_addr += NOC_MAX_BURST_SIZE;
        dst_local_l1_addr += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
    }
    cq_noc_async_read_with_trid(src_addr, dst_local_l1_addr, size, trid);
}

FORCE_INLINE
void cq_noc_async_read_barrier_with_trid(uint32_t trid) {
    noc_async_read_barrier_with_trid(trid);
}

FORCE_INLINE
void noc_helper(uint32_t src, uint64_t dst, uint32_t len_bytes) {
    while (!noc_cmd_buf_ready(0, NCRISC_WR_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(0, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[0] += 1;
    noc_nonposted_writes_acked[0] += 1;
}

FORCE_INLINE
void noc_helper_any_len(uint32_t src, uint64_t dst, uint32_t len_bytes) {
    while (len_bytes > NOC_MAX_BURST_SIZE) {
        noc_helper(src, dst, NOC_MAX_BURST_SIZE);
        src += NOC_MAX_BURST_SIZE;
        dst += NOC_MAX_BURST_SIZE;
        len_bytes -= NOC_MAX_BURST_SIZE;
    }
    noc_helper(src, dst, len_bytes);
}

void kernel_main() {
#ifdef PAGE_SIZE
    uint32_t page_size = PAGE_SIZE;
#else
    uint32_t page_size = get_arg_val<uint32_t>(0);
#endif
    cb_reserve_back(0, PAGE_COUNT);
    uint64_t downstream_noc_addr = NOC_XY_ADDR(NOC_X(2), NOC_Y(4), get_write_ptr(0));

    uint32_t noc_cmd_field =
    NOC_CMD_CPY | NOC_CMD_WR |
    NOC_CMD_VC_STATIC  |
    NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)downstream_noc_addr);
    NOC_CMD_BUF_WRITE_REG(0, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, get_write_ptr(0));
    NOC_CMD_BUF_WRITE_REG(0, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(downstream_noc_addr >> NOC_ADDR_COORD_SHIFT));
    NOC_CMD_BUF_WRITE_REG(0, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE, NOC_MAX_BURST_SIZE);
    for (int it = 0; it < ITERATIONS; it++) {
        uint32_t cb_addr = get_write_ptr(0);
        uint32_t ring_db_slot_size = scratch_db_half_size; // page_size << 8;
        ring_db_slot_size = ring_db_slot_size > scratch_db_half_size ? ((scratch_db_half_size / page_size) * page_size) : ring_db_slot_size;
        // noc_async_write_one_packet_set_state(downstream_noc_addr, ring_db_slot_size);
        // DPRINT << ring_db_slot_size << ENDL();
        uint32_t num_slots = scratch_db_size / ring_db_slot_size;
        num_slots = (num_slots > 13) ? 13 : num_slots;
        uint32_t ring_db_size = num_slots * ring_db_slot_size;

        uint32_t read_length = PAGE_COUNT * page_size;
        uint32_t ring_db_read_addr = cb_addr;
        uint32_t amt_to_read = (ring_db_size > read_length) ? read_length : ring_db_size;
        read_length -= amt_to_read;
        uint32_t amt_to_write = amt_to_read;
        // DPRINT << read_length << " " << amt_to_read << " " << amt_to_write <<  ENDL();
        int page_id = 0;
        while (amt_to_read) {
            uint32_t trid = (ring_db_read_addr - cb_addr) / (ring_db_slot_size) + 1;
            uint64_t noc_addr = get_dram_noc_addr(page_id, page_size, 0);
            while(!noc_cmd_buf_ready(1, NCRISC_RD_CMD_BUF));
            ncrisc_noc_set_transaction_id(1, NCRISC_RD_CMD_BUF, trid);
            NOC_CMD_BUF_WRITE_REG(1, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);
            cq_noc_async_read_with_trid_any_len(noc_addr, ring_db_read_addr, page_size, trid);
            ring_db_read_addr += page_size;
            page_id++;
            amt_to_read -= page_size;
        }
        noc_reads_num_issued[1] += amt_to_read / page_size;
        uint32_t write_slot = 0;
        uint32_t ring_db_write_addr = cb_addr;
        while (read_length > 0 || amt_to_write > ring_db_slot_size) {
            // DPRINT << "In loop" << ENDL();
            uint32_t trid = write_slot + 1;
            amt_to_write -= ring_db_slot_size;
            cq_noc_async_read_barrier_with_trid(trid);
            noc_helper_any_len(ring_db_write_addr, downstream_noc_addr, ring_db_slot_size);
            if (read_length) {
                amt_to_read =  (ring_db_slot_size > read_length) ? read_length : ring_db_slot_size;
                amt_to_write += amt_to_read;
                read_length -= amt_to_read;
                ring_db_read_addr = ring_db_write_addr;
                while(!noc_cmd_buf_ready(1, NCRISC_RD_CMD_BUF));
                ncrisc_noc_set_transaction_id(1, NCRISC_RD_CMD_BUF, trid);
                NOC_CMD_BUF_WRITE_REG(1, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, page_size);
                noc_async_writes_flushed();
                while (amt_to_read) {
                    uint64_t noc_addr = get_dram_noc_addr(page_id, page_size, 0);
                    cq_noc_async_read_with_trid_any_len(noc_addr, ring_db_read_addr, page_size, trid);
                    ring_db_read_addr += page_size;
                    page_id++;
                    amt_to_read -= page_size;
                }
                noc_reads_num_issued[1] += amt_to_read / page_size;
                write_slot = trid % num_slots;
                ring_db_write_addr = cb_addr + write_slot * ring_db_slot_size;
            }
        }
        cq_noc_async_read_barrier_with_trid(write_slot + 1);
        noc_helper_any_len(ring_db_write_addr, downstream_noc_addr, ring_db_slot_size);
    }
}
