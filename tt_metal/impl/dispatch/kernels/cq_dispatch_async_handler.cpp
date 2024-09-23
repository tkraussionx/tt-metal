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

constexpr uint32_t cb_base = get_compile_time_arg_val(0);
constexpr uint32_t cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t cb_size = get_compile_time_arg_val(2);
constexpr uint32_t my_dispatch_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t upstream_dispatch_cb_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_s_sync_sem_id = get_compile_time_arg_val(5);
constexpr uint32_t worker_mcast_grid = get_compile_time_arg_val(6);
constexpr uint32_t num_worker_cores_to_mcast = get_compile_time_arg_val(7);
constexpr uint32_t mcast_go_signal_addr = get_compile_time_arg_val(8);
constexpr uint32_t unicast_go_signal_addr = get_compile_time_arg_val(9);
constexpr uint32_t distributed_dispatcher = get_compile_time_arg_val(10); // dispatch_s and dispatch_d running on different cores
constexpr uint32_t worker_sem_addr = get_compile_time_arg_val(11); // workers update the semaphore at this location to signal completion

constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t dispatch_d_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint8_t my_noc_index = NOC_INDEX;

constexpr uint32_t cb_page_size = 1 << cb_log_page_size;
constexpr uint32_t cb_end = cb_base + cb_size;

static uint32_t num_pages_acquired = 0;
static uint32_t num_mcasts_sent = 0;
static uint32_t curr_num_workers_completed = 0;
static uint32_t cmd_ptr;
static uint32_t unicast_only_cores[16]; // Allocate this on stack
static int num_unicast_cores = -1;
constexpr uint8_t send_mcast = 0x1; // Make enum
constexpr uint8_t send_unicast = 0x2;

// Cleanup noc init (dont explicitly init NOC 1 and setup cmd_buf fields for brisc_at_cmd_buf in this kernel)
// Cleanup wrap based checking

// Initialize the go_signal data that will be sent to workers over NOC1 in L1
uint32_t aligned_go_signal __attribute__((aligned(16))) __attribute__((section("l1_data"))) __attribute__((used)) = RUN_MSG_GO;

FORCE_INLINE
uint32_t wrapped_distance(uint32_t num_pages_released, uint32_t num_pages_acquired) {
    // num_pages_released >= num_pages_acquired at all times
    return (num_pages_released > num_pages_acquired) ? (num_pages_released - num_pages_acquired) : (UINT32_MAX - num_pages_acquired + num_pages_released + 1);
}

FORCE_INLINE
void wait_for_workers(volatile CQDispatchCmd tt_l1_ptr *cmd) {
    volatile tt_l1_ptr uint32_t* worker_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr);
    while (*worker_sem < cmd->mcast.wait_count);
}

FORCE_INLINE
void update_worker_completion_count_on_dispatch_d() {
    if constexpr(distributed_dispatcher) {
        uint32_t num_workers_signalling_completion = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr);
        if (num_workers_signalling_completion != curr_num_workers_completed) {
            curr_num_workers_completed = num_workers_signalling_completion;
            uint64_t dispatch_d_dst = get_noc_addr_helper(dispatch_d_noc_xy, worker_sem_addr);
            noc_async_write_one_packet(worker_sem_addr, dispatch_d_dst, sizeof(uint32_t));
        }
    }
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
        update_worker_completion_count_on_dispatch_d();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("DAPD");
    num_pages_acquired += n;
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    // Get semaphore that will be update by dispatch_d, signalling that its safe to send a go signal
    volatile tt_l1_ptr uint32_t* sync_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(dispatch_s_sync_sem_id));
    aligned_go_signal = cmd->mcast.go_signal; // Copy the go signal from the command to a NOC Aligned L1 Location

    // Wait for notification from dispatch_d, signalling that its safe to send the go signal
    while (*sync_sem_addr <= num_mcasts_sent) {
        // Update dispatch_d with the latest num_workers
        update_worker_completion_count_on_dispatch_d();
    }
    num_mcasts_sent++; // Go signal sent -> update counter
    // Wait until workers have completed before sending go signal
    wait_for_workers(cmd);
    // send go signal update here
    if (cmd->mcast.mcast_flag & send_mcast) {
        uint64_t dst = get_noc_addr_helper(worker_mcast_grid, mcast_go_signal_addr);
        noc_async_write_multicast_one_packet((uint32_t)(&aligned_go_signal), dst, sizeof(uint32_t), num_worker_cores_to_mcast);
    }
    if (cmd->mcast.mcast_flag & send_unicast) {
        // If dispatch_s needs to unicast the go signal to specific cores, num_unicast_cores
        // must be set using set_go_signal_unicast_only_cores
        ASSERT(num_unicast_cores > 0);
        for (int core_idx = 0; core_idx < num_unicast_cores; core_idx++) {
            uint64_t dst = get_noc_addr_helper(unicast_only_cores[core_idx], unicast_go_signal_addr);
            noc_async_write_one_packet((uint32_t)(&aligned_go_signal), dst, sizeof(uint32_t));
        }
    }
    update_worker_completion_count_on_dispatch_d();
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_go_signal_unicast_only_cores() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    num_unicast_cores = (int)(cmd->set_unicast_only_cores.num_unicast_only_cores);
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    for (int core_idx = 0; core_idx < num_unicast_cores; core_idx++) {
        DPRINT << "Unicast encoding: " << *((uint32_t tt_l1_ptr*)data_ptr) << ENDL();
        unicast_only_cores[core_idx] = *((uint32_t tt_l1_ptr*)data_ptr);
        data_ptr += sizeof(uint32_t);
    }
    cmd_ptr += sizeof(CQDispatchCmd) + num_unicast_cores * sizeof(uint32_t);
}

FORCE_INLINE
void process_dispatch_s_wait_cmd() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    // Limited Usage of Wait CMD: dispatch_s should get a wait command only if its not on the
    // same core as dispatch_d and is used to clear the worker count
    ASSERT(cmd->wait.clear_count && (cmd->wait.addr == worker_sem_addr) && distributed_dispatcher);
    volatile tt_l1_ptr uint32_t* worker_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr);
    // Wait for workers to complete
    while (*worker_sem < cmd->wait.count);
    // Send updated worker count to dispatch_d
    update_worker_completion_count_on_dispatch_d();
    // Wait for updated count to get written and then clear the counter.
    // dispatch_d will clear its own counter
    while(!ncrisc_noc_nonposted_writes_flushed(1));
    *worker_sem = 0;
}

void kernel_main() {
    // DPRINT << "Dispatch Handler Started: " << cb_base  << " " << cb_end << ENDL();
    noc_local_state_init(1);
    cmd_ptr = cb_base;
    bool done = false;
    while(!done) {
        // These need to use NOC 1 BRISC_AT_CMD_BUF
        // DPRINT << "Trying to acquire pages: " << cmd_ptr << " " << cb_base << " " << cb_end << ENDL();
        // Send go signal here (idling). This is okay for now, since dispatch_s can only proceed once dispatch_d sees workers as completed
        // Need a clear count signal (that needs to wait)
        cb_acquire_pages_dispatch_s<my_noc_xy, my_dispatch_cb_sem_id>(1);

        volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
        switch (cmd->base.cmd_id) {
            case CQ_DISPATCH_CMD_GO_SIGNAL_MCAST:
                process_go_signal_mcast_cmd();
                break;
            case CQ_DISPATCH_SET_UNICAST_ONLY_CORES:
                set_go_signal_unicast_only_cores();
                break;
            case CQ_DISPATCH_CMD_WAIT:
                process_dispatch_s_wait_cmd();
                break;
            case CQ_DISPATCH_CMD_TERMINATE:
                done = true;
                break;
            default:
                DPRINT << "dispatcher_s invalid command" << ENDL();
                ASSERT(0);
        }
        cmd_ptr = round_up_pow2(cmd_ptr, cb_page_size);
        // Release a single page to prefetcher. Assumption is that all dispatch_s commands fit inside a single page for now.
        cb_release_pages<my_noc_index, upstream_noc_xy, upstream_dispatch_cb_sem_id>(1);
        if (cmd_ptr == cb_end) {
            cmd_ptr = cb_base;
        }
    }
}
