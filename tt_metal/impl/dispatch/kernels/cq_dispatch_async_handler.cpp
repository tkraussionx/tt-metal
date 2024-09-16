// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch kernel
//  - receives data in pages from prefetch kernel into the dispatch buffer ring buffer
//  - processes commands with embedded data from the dispatch buffer to write/sync/etc w/ destination
//  - sync w/ prefetcher is via 2 semaphores, page_ready, page_done
//  - page size must be a power of 2
//  - # blocks must evenly divide the dispatch buffer size
//  - dispatch buffer base must be page size aligned

#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t cb_base = get_compile_time_arg_val(22);
constexpr uint32_t cb_size = get_compile_time_arg_val(30);
constexpr uint32_t cb_end = cb_base + cb_size;

constexpr uint32_t my_dispatch_cb_sem_id = get_compile_time_arg_val(23);
constexpr uint32_t upstream_dispatch_cb_sem_id = get_compile_time_arg_val(24);
constexpr uint32_t dispatch_s_sync_sem_id = get_compile_time_arg_val(25);
constexpr uint32_t worker_mcast_grid = get_compile_time_arg_val(26);
constexpr uint32_t num_worker_cores_to_mcast = get_compile_time_arg_val(27);
constexpr uint32_t mcast_go_signal_addr = get_compile_time_arg_val(28);
constexpr uint32_t unicast_go_signal_addr = get_compile_time_arg_val(29);

// Get dispatch_s coords from RTA. TODO: When dispatch_s is separated out, get this from CTAs.
// Upstream coords will need to explicitly be programmed in CTA as well
// dispatch_d and prefetch_hd need to have specific CTA for dispatch_s coords
constexpr uint32_t my_noc_xy = get_compile_time_arg_val(31); // uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));

static uint32_t num_pages_acquired = 0;
static uint32_t num_mcasts_sent = 0;

// Initialize the go_signal data that will be sent to workers over NOC1 in L1
uint32_t aligned_go_signal __attribute__((aligned(16))) __attribute__((section("l1_data"))) __attribute__((used)) = RUN_MSG_GO;

FORCE_INLINE
void dispatch_s_noc_semaphore_inc(uint64_t addr, uint32_t incr, uint8_t noc_id = noc_index) {
    /*
    [REFER TO grayskull/noc/noc.h for the documentation of noc_atomic_increment()]
    Generic increment with 32-bit wrap.
  */
    WAYPOINT("NSIW");
    DEBUG_SANITIZE_NOC_ADDR(noc_id, addr, 4);
    DEBUG_INSERT_DELAY(TransactionAtomic);
    noc_fast_atomic_increment(noc_id, BRISC_AT_CMD_BUF, addr, NOC_UNICAST_WRITE_VC, incr, 31 /*wrap*/, false /*linked*/, false /*posted*/);
    WAYPOINT("NSID");
}

FORCE_INLINE
uint32_t wrapped_distance(uint32_t num_pages_released, uint32_t num_pages_acquired) {
    // num_pages_released >= num_pages_acquired at all times
    return (num_pages_released > num_pages_acquired) ? (num_pages_released - num_pages_acquired) : (UINT32_MAX - num_pages_acquired + num_pages_released + 1);
}

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_acquire_pages_dispatch_s(uint32_t n) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    WAYPOINT("DAPW");
    // Use a wrapping compare here to compare distance
    // Required for trace which steals downstream credits and may make the value negative
    uint32_t heartbeat = 0;
    // DPRINT << " Num Pages acquired: " << num_pages_acquired << ENDL();
    // DPRINT <<  "Num pages release: " << *sem_addr << ENDL();
    while (wrapped_distance(*sem_addr, num_pages_acquired) < n) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("DAPD");
    num_pages_acquired += n;
}

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_release_pages_dispatch_s(uint32_t n) {
    dispatch_s_noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), n, 1);
}

FORCE_INLINE
void noc_async_write_multicast_one_packet_dispatch_s(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    bool multicast_path_reserve = true) {
    WAYPOINT("NMPW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(1, dst_noc_addr_multicast, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(1, NCRISC_WR_CMD_BUF));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field =
                            NOC_CMD_CPY | NOC_CMD_WR |
                            NOC_CMD_VC_STATIC |
                            NOC_CMD_STATIC_VC(NOC_MULTICAST_WRITE_VC) |
                            (linked ? NOC_CMD_VC_LINKED : 0x0) |
                            ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) |
                            NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr_multicast);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr_multicast >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr_multicast >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE,  size);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[1] += 1;
    noc_nonposted_writes_acked[1] += num_dests;
}

FORCE_INLINE
void noc_async_write_unicast_one_packet_dispatch_s(std::uint32_t src_local_l1_addr, std::uint64_t dst_noc_addr, std::uint32_t size) {
    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(1, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(1, NCRISC_WR_CMD_BUF));
    WAYPOINT("NWPD");

    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 0x0 |  // (linked ? NOC_CMD_VC_LINKED : 0x0)
                                0x0 |  // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
                                NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_local_l1_addr);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dst_noc_addr);
#ifdef ARCH_BLACKHOLE
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & 0x1000000F);
#endif
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE,  size);
    NOC_CMD_BUF_WRITE_REG(1, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[1] += 1;
    noc_nonposted_writes_acked[1] += 1;  // num_dests
}

void kernel_main() {
    // DPRINT << "Dispatch Handler Started: " << cb_base  << " " << cb_end << ENDL();
    noc_local_state_init(1);
    uint32_t cmd_ptr = cb_base;
    bool done = false;
    uint32_t unicast_only_cores[16];
    int num_unicast_cores = -1;
    uint8_t send_mcast = 0x1;
    uint8_t send_unicast = 0x2;
    while(!done) {
        // These need to use NOC 1 BRISC_AT_CMD_BUF
        // DPRINT << "Trying to acquire pages: " << cmd_ptr << " " << cb_base << " " << cb_end << ENDL();
        cb_acquire_pages_dispatch_s<my_noc_xy, my_dispatch_cb_sem_id>(1);
        // DPRINT << "Acquired pages" << ENDL();
        volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
        if (cmd->base.cmd_id == CQ_DISPATCH_CMD_GO_SIGNAL_MCAST) {
            // DPRINT << "Received Mcast Command " << num_worker_cores_to_mcast  << " " << cmd_ptr <<  ENDL();
            volatile tt_l1_ptr uint32_t* sync_sem_addr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(dispatch_s_sync_sem_id));
            volatile tt_l1_ptr uint32_t* worker_sem_addr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cmd->mcast.wait_addr);
            uint64_t dst = get_noc_addr_helper(worker_mcast_grid, mcast_go_signal_addr);

            // Wait for notification from dispatch_d, signalling that its safe to send the go signal
            while (*sync_sem_addr <= num_mcasts_sent);
            // Wait until workers have completed before sending go signal
            while (*worker_sem_addr < cmd->mcast.wait_count);
            num_mcasts_sent++;
            aligned_go_signal = cmd->mcast.go_signal;
            if (cmd->mcast.mcast_flag & send_mcast) {
                // DPRINT << " Go Signal " << (cmd->mcast.go_signal & 0xFF) << " " <<  (cmd->mcast.go_signal & 0xFF00) << " " << (cmd->mcast.go_signal & 0xFF0000) << ENDL();
                noc_async_write_multicast_one_packet_dispatch_s((uint32_t)(&aligned_go_signal), dst, sizeof(uint32_t), num_worker_cores_to_mcast);
            }
            if (cmd->mcast.mcast_flag & send_unicast) {
                for (int core_idx = 0; core_idx < num_unicast_cores; core_idx++) {
                    uint64_t dst = get_noc_addr_helper(unicast_only_cores[core_idx], unicast_go_signal_addr);
                    noc_async_write_unicast_one_packet_dispatch_s((uint32_t)(&aligned_go_signal), dst, sizeof(uint32_t));
                }
            }
            while(!ncrisc_noc_nonposted_writes_flushed(1));
            // DPRINT << "Done write" << ENDL();
        }
        else if (cmd->base.cmd_id == CQ_DISPATCH_CMD_TERMINATE) {
            // DPRINT << "dispatch_s Terminating" << ENDL();
            done = true;
        }
        else if (cmd->base.cmd_id == CQ_DISPATCH_SET_UNICAST_ONLY_CORES) {
            DPRINT << "Recieved CQ_DISPATCH_SET_UNICAST_ONLY_CORES cmd" << ENDL();
            num_unicast_cores = (int)(cmd->set_unicast_only_cores.num_unicast_only_cores);
            uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);;
            for (int core_idx = 0; core_idx < num_unicast_cores; core_idx++) {
                DPRINT << "Unicast encoding: " << *((uint32_t tt_l1_ptr*)data_ptr) << ENDL();
                unicast_only_cores[core_idx] = *((uint32_t tt_l1_ptr*)data_ptr);
                data_ptr += sizeof(uint32_t);
            }
            cmd_ptr += num_unicast_cores * sizeof(uint32_t);
        }
        else {
            DPRINT << "Got invalid command" << ENDL();
        }
        cmd_ptr += sizeof(CQDispatchCmd);
        cmd_ptr = round_up_pow2(cmd_ptr, dispatch_cb_page_size);
        cb_release_pages_dispatch_s<upstream_noc_xy, upstream_dispatch_cb_sem_id>(1);
        if (cmd_ptr == cb_end) {
            cmd_ptr = cb_base;
        }
    }
}
