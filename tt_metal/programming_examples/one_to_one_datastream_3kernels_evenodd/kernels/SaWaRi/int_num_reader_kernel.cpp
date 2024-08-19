// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
	uint32_t dram_buffer  = get_arg_val<uint32_t>(0);
	uint32_t semaphore_odd_local_addr  = get_semaphore(get_arg_val<uint32_t>(1));
	uint32_t semaphore_even_local_addr = get_semaphore(get_arg_val<uint32_t>(2));
	uint32_t semaphore_read_local_addr = get_semaphore(get_arg_val<uint32_t>(3));
	uint32_t core_odd_x_cord = get_arg_val<uint32_t>(4);
	uint32_t core_odd_y_cord = get_arg_val<uint32_t>(5);
	uint32_t core_even_x_cord = get_arg_val<uint32_t>(6);
	uint32_t core_even_y_cord = get_arg_val<uint32_t>(7);
	uint32_t core_read_x_cord = get_arg_val<uint32_t>(8);
	uint32_t core_read_y_cord = get_arg_val<uint32_t>(9);
	uint32_t dram_noc_x = get_arg_val<uint32_t>(10);
	uint32_t dram_noc_y = get_arg_val<uint32_t>(11);

	// Fetch the semaphore address converted for NOC operations
	uint64_t semaphore_odd_noc_addr = get_noc_addr(core_odd_x_cord, core_odd_y_cord, semaphore_odd_local_addr);
	uint64_t semaphore_even_noc_addr = get_noc_addr(core_even_x_cord, core_even_y_cord, semaphore_even_local_addr);
	volatile tt_l1_ptr uint32_t* semaphore_read_l1_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_read_local_addr);

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

 	noc_semaphore_inc(semaphore_odd_noc_addr, 1);	 // signals to generator kernel to write value to DRAM

    // Read data stream
	for (int integer = 1; integer < 1001; integer++) {

		noc_semaphore_wait(semaphore_read_l1_address, 1);

		// Read data from DRAM to the TT-CircularBuffer cb_read
		noc_async_read(dram_noc_addr, l1_read_addr, 4);

 		noc_semaphore_set(semaphore_read_l1_address, 0); // resets reader kernel's semaphore so that on next iteration in this for loop, reader waits for generator to write next value in stream to dram

		if (integer % 2)
			noc_semaphore_inc(semaphore_even_noc_addr, 1); // tells generator kernel to write next EVEN value in stream to dram
		else
			noc_semaphore_inc(semaphore_odd_noc_addr, 1); // tells generator kernel to write next ODD value in stream to dram

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

	DPRINT << "SaWaRi" << ENDL();
	DPRINT << "InSequence = " << in_sequence << ENDL();
	DPRINT << "EqualValue = " << equal_value << ENDL();
	DPRINT << "InOrder = " << in_order << ENDL();
	DPRINT << "OutOfOrder = " << out_of_order << ENDL();
}
