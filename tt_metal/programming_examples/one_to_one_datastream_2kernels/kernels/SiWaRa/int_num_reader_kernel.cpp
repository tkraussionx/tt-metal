// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
	uint32_t dram_buffer  = get_arg_val<uint32_t>(0);
	uint32_t core_x_cord = get_arg_val<uint32_t>(3);
	uint32_t core_y_cord = get_arg_val<uint32_t>(4);
	uint32_t dram_noc_x = get_arg_val<uint32_t>(5);
	uint32_t dram_noc_y = get_arg_val<uint32_t>(6);

	uint64_t dram_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_buffer);

	constexpr uint32_t cb_read = tt::CB::c_in1; // index=1
	uint32_t l1_read_addr = get_write_ptr(cb_read);

	// Statistics parameters initialization
	int in_sequence = 0;
	int equal_value = 0;
	int in_order = 0;
	int out_of_order = 0;

	// Starting value
	int old_read_value = 0;
	int new_read_value = 0;
	uint32_t* read_ptr = (uint32_t*) l1_read_addr;
	read_ptr = 0;

    // Read data stream
	for (int integer = 1; integer < 1001; integer++) {

		// Read data from DRAM to the TT-CircularBuffer cb_read
		noc_async_read(dram_noc_addr, l1_read_addr, 4);
		noc_async_read_barrier();

 		uint32_t* read_ptr = (uint32_t*) l1_read_addr;
		new_read_value = *read_ptr;

		if (old_read_value + 1 == new_read_value) {
			in_sequence += 1;
		}
		if (old_read_value == new_read_value) {
			equal_value += 1;
		}
		if (old_read_value < new_read_value) {
			in_order += 1;
		}
		if (old_read_value > new_read_value) {
			out_of_order += 1;
		}

		old_read_value = new_read_value;
	}

	DPRINT << "SiWaRa" << ENDL();
	DPRINT << "InSequence = " << in_sequence << ENDL();
	DPRINT << "EqualValue = " << equal_value << ENDL();
	DPRINT << "InOrder = " << in_order << ENDL();
	DPRINT << "OutOfOrder = " << out_of_order << ENDL();
}
