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
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>



CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

void kernel_launch() {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    kernel_main();
}
