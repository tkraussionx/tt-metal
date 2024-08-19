// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
	uint32_t dram_buffer  = get_arg_val<uint32_t>(0);
	uint32_t core_gen_x_cord = get_arg_val<uint32_t>(3);
	uint32_t core_gen_y_cord = get_arg_val<uint32_t>(4);
	uint32_t core_read_x_cord = get_arg_val<uint32_t>(5);
	uint32_t core_read_y_cord = get_arg_val<uint32_t>(6);
	uint32_t dram_noc_x = get_arg_val<uint32_t>(7);
	uint32_t dram_noc_y = get_arg_val<uint32_t>(8);

	uint64_t dram_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_buffer);

	constexpr uint32_t cb_write = tt::CB::c_in0; // index=0

	uint32_t l1_write_addr = get_write_ptr(cb_write);
    uint32_t* write_ptr;

	// Write stream
	for (int integer = 1; integer < 1001; integer++) {

		// Write the integer value to the TT-CircularBuffer cb_write
		write_ptr = (uint32_t*) l1_write_addr;
		*write_ptr = integer;

		// Write data from TT-CircularBuffer cb_write to DRAM
		noc_async_write(l1_write_addr, dram_noc_addr, 4);

	}
}
