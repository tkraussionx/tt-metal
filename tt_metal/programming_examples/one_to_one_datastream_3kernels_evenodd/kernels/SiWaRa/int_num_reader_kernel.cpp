// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
	uint32_t dram_buffer  = get_arg_val<uint32_t>(0);
	uint32_t core_odd_x_cord = get_arg_val<uint32_t>(4);
	uint32_t core_odd_y_cord = get_arg_val<uint32_t>(5);
	uint32_t core_even_x_cord = get_arg_val<uint32_t>(6);
	uint32_t core_even_y_cord = get_arg_val<uint32_t>(7);
	uint32_t core_read_x_cord = get_arg_val<uint32_t>(8);
	uint32_t core_read_y_cord = get_arg_val<uint32_t>(9);
	uint32_t dram_noc_x = get_arg_val<uint32_t>(10);
	uint32_t dram_noc_y = get_arg_val<uint32_t>(11);

	uint64_t dram_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_buffer);

	constexpr uint32_t cb_read = tt::CB::c_in0; // index=0
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

		// Read the value from the TT-CircularBuffer cb_read
 		uint32_t* read_ptr = (uint32_t*) l1_read_addr;
		new_read_value = *read_ptr;

		// Check if values are in sequence, equal, etc.
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
