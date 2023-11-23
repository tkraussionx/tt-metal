// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "noc_parameters.h"
#include "ethernet/dataflow_api.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "risc_attribs.h"
#include "tensix.h"
#include "tensix_types.h"
#include "tt_eth_api.h"
#include "c_tensix_core.h"
#include "ethernet/noc_nonblocking_api.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"
#include "noc_addr_ranges_gen.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_common.h"
#include <kernel.cpp>


CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
CQReadInterface cq_read_interface;

void ApplicationHandler(void) __attribute__((__section__(".init")));


uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_xy_local_addr[NUM_NOCS] __attribute__((used));
uint8_t noc_index = NOC_INDEX;
void erisc_init() {
      for (uint32_t n = 0; n < NUM_NOCS; n++) {
      uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
      my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
      my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

       // set_noc_trans_table(n, noc_trans_table_en, my_logical_x[n], my_logical_y[n]);
  }
}

void __attribute__((__section__("erisc_l1_code"))) kernel_launch() {
    RISC_POST_STATUS(0x12345678);
    kernel_main();
    RISC_POST_STATUS(0x12345679);
    disable_erisc_app();
}
void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    ncrisc_noc_init();

    erisc_init();

    setup_cb_read_write_interfaces(0, 1, true, true);
    //noc_local_state_init(noc_index);
    // TODO:: if bytes done not updated, yield to task 1
    kernel_launch();
}
