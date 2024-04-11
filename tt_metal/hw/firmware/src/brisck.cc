// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
#include "noc_addr_ranges_gen.h"

#include <kernel.cpp>

#include "debug/status.h"
#include "debug/dprint.h"

uint8_t noc_index = NOC_INDEX;

// const uint32_t read_cmd_buf __attribute__((used)) = BRISC_RD_CMD_BUF;
// const uint32_t write_cmd_buf __attribute__((used)) = BRISC_WR_REG_CMD_BUF;
// const uint32_t read_transaction_id __attribute__((used)) = BRISC_RD_TRANSACTION_ID;
// const uint32_t write_transaction_id __attribute__((used)) = BRISC_WR_REG_TRANSACTION_ID;
// const uint32_t use_multi_noc __attribute__((used)) = false;
// const uint32_t noc_index_to_dram_bank_map[NUM_DRAM_BANKS] __attribute__((used)) = {
//     NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX,
//     1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX, 1 - NOC_INDEX
// };

// const uint32_t noc_index_to_dram_bank_map[NUM_DRAM_BANKS] __attribute__((used)) = {
//     NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX,
//     NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX, NOC_INDEX };


void kernel_launch() {

#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    firmware_kernel_common_init((void tt_l1_ptr *)MEM_BRISC_INIT_LOCAL_L1_BASE);

    // noc_local_state_init(noc_index);
    noc_local_state_init(0);
    noc_local_state_init(1);

    // DPRINT << noc_reads_num_issued[0] << ENDL();
    // DPRINT << noc_reads_num_issued[1] << ENDL();

    {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        kernel_main();
    }
#endif
}
