#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "ckernel_globals.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdevcommon/profiler_common.h"
#include "dataflow_api.h"
#include "debug_print.h"

#include <kernel.cpp>

#include "debug_print.h"

CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];
CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
CQReadInterface cq_read_interface;

#ifdef NOC_INDEX
uint8_t loading_noc = NOC_INDEX;
#else
uint8_t loading_noc = 0;
#endif

volatile uint32_t * const ncrisc_run_mailbox_address =
    (volatile uint32_t *)(MEM_RUN_MAILBOX_ADDRESS + MEM_MAILBOX_NCRISC_OFFSET);
volatile uint32_t * const trisc_run_mailbox_addresses[3] = {
    (volatile uint32_t *)(MEM_RUN_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC0_OFFSET),
    (volatile uint32_t *)(MEM_RUN_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC1_OFFSET),
    (volatile uint32_t *)(MEM_RUN_MAILBOX_ADDRESS + MEM_MAILBOX_TRISC2_OFFSET)
};

// to reduce the amount of code changes, both BRISC and NRISCS instatiate these counter for both NOCs (ie NUM_NOCS)
// however, atm NCRISC uses only NOC-1 and BRISC uses only NOC-0
// this way we achieve full separation of counters / cmd_buffers etc.
uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];

// dram channel to x/y lookup tables
// The number of banks is generated based off device we are running on --> controlled by allocator
uint8_t dram_bank_to_noc_x[NUM_DRAM_BANKS];
uint8_t dram_bank_to_noc_y[NUM_DRAM_BANKS];
uint32_t dram_bank_to_noc_xy[NUM_DRAM_BANKS];

uint8_t l1_bank_to_noc_x[NUM_L1_BANKS];
uint8_t l1_bank_to_noc_y[NUM_L1_BANKS];
uint32_t l1_bank_to_noc_xy[NUM_L1_BANKS];

extern uint64_t dispatch_addr;
extern uint8_t kernel_noc_id_var;

inline void notify_host_kernel_finished() {
    uint32_t pcie_noc_x = NOC_X(0);
    uint32_t pcie_noc_y = NOC_Y(4); // These are the PCIE core coordinates
    uint64_t pcie_address =
        get_noc_addr(pcie_noc_x, pcie_noc_y, 0);  // For now, we are writing to host hugepages at offset 0 (nothing else currently writing to it)

    volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(NOTIFY_HOST_KERNEL_COMPLETE_ADDR);
    done[0] = NOTIFY_HOST_KERNEL_COMPLETE_VALUE; // 512 was chosen arbitrarily, but it's less common than 1 so easier to check validity

    // Write to host hugepages to notify of completion
    noc_async_write(NOTIFY_HOST_KERNEL_COMPLETE_ADDR, pcie_address, 4);
    noc_async_write_barrier();
}

inline __attribute__((always_inline)) void finish_BR_profiler()
{
#if defined(PROFILE_KERNEL) && defined(COMPILE_FOR_BRISC)

    const uint32_t NOC_ID_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1;
    uint32_t noc_id = noc_local_node_id() & 0xFFF;
    uint32_t dram_noc_x = noc_id & NOC_ID_MASK;
    uint32_t dram_noc_y = (noc_id >> NOC_ADDR_NODE_ID_BITS) & NOC_ID_MASK;

    uint32_t core_flat_id;
    constexpr int DRAM_ROW = 6;
    if (dram_noc_y > DRAM_ROW){
        core_flat_id = l1_buffer_count*((dram_noc_y - 2) * 12 + (dram_noc_x - 1));
    }
    else{
        core_flat_id = l1_buffer_count*((dram_noc_y - 1) * 12 + (dram_noc_x - 1));
    }

    volatile uint32_t *buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_NC);
    uint32_t page_id = buffer[kernel_profiler::PAGE_COUNTER];
    buffer[kernel_profiler::PAGE_COUNTER]++;

    uint32_t dram_address = DRAM_PROFILER_ADDRESS + (core_flat_id + page_id) * l1_buffer_size;

    //volatile uint32_t *debug_buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_NC);
    //debug_buffer[0] = dram_noc_x;
    //debug_buffer[1] = dram_noc_y;
    //debug_buffer[2] = core_flat_id;
    //debug_buffer[3] = dram_address;
    //debug_buffer[4] = page_id;


    if (page_id < l1_buffer_count)
    {
        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(1, 0, dram_address);
        noc_async_write(PRINT_BUFFER_NC, dram_buffer_dst_noc_addr, l1_buffer_size);
        noc_async_write_barrier();
    }
#endif //PROFILE_KERNEL
}

void kernel_launch() {

    firmware_kernel_common_init((void *)MEM_BRISC_INIT_LOCAL_L1_BASE);

#if defined(IS_DISPATCH_KERNEL)
    setup_cq_read_write_interface();
#else
    setup_cb_read_write_interfaces();                // done by both BRISC / NCRISC
#endif

    init_dram_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC
    init_l1_bank_to_noc_coord_lookup_tables();  // done by both BRISC / NCRISC

    noc_init(loading_noc);

    kernel_main();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & KERNEL_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif

    volatile uint32_t* use_triscs = (volatile uint32_t*)(MEM_ENABLE_TRISC_MAILBOX_ADDRESS);
    if (*use_triscs) {
        while (
            !(*trisc_run_mailbox_addresses[0] == 1 &&
              *trisc_run_mailbox_addresses[1] == 1 &&
              *trisc_run_mailbox_addresses[2] == 1)) {
        }

        // Once all 3 have finished, assert reset on all of them
        assert_trisc_reset();
    }

    volatile uint32_t* use_ncrisc = (volatile uint32_t*)(MEM_ENABLE_NCRISC_MAILBOX_ADDRESS);
    if (*use_ncrisc) {
        while (*ncrisc_run_mailbox_address != 1);
    }

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
    kernel_profiler::mark_time(CC_MAIN_END);
    finish_BR_profiler();
#endif

    // FW needs to notify device dispatcher when we are done
    // Some information needed is known here, pass it back
    kernel_noc_id_var = loading_noc;
#if defined(TT_METAL_DEVICE_DISPATCH_MODE)
    dispatch_addr = (my_x[loading_noc] == NOC_X(1) && my_y[loading_noc] == NOC_Y(11)) ?
        0 :
        get_noc_addr(1, 11, DISPATCH_MESSAGE_ADDR);
#else
    dispatch_addr = 0;
#endif

}
