
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
  uint32_t arg = 0;
  const auto input0_addr = get_arg_val<uint32_t>(arg++);
  const auto input1_addr = get_arg_val<uint32_t>(arg++);
  const auto input0_cb = get_arg_val<uint32_t>(arg++);
  const auto input1_cb = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);
  const auto tile_offset = get_arg_val<uint32_t>(arg++);

  constexpr bool input0_is_dram = get_compile_time_arg_val(0) == 1;
  constexpr bool input1_is_dram = get_compile_time_arg_val(0) == 1;

  const uint32_t input0_page_size = get_tile_size(input0_cb);
  const auto input0_data_format = get_dataformat(input0_cb);
  const InterleavedAddrGenFast<input0_is_dram> input0_addrg = {
      .bank_base_address = input0_addr,
      .page_size = input0_page_size,
      .data_format = input0_data_format};

  const uint32_t input1_page_size = get_tile_size(input1_cb);
  const auto input1_data_format = get_dataformat(input1_cb);
  const InterleavedAddrGenFast<input1_is_dram> input1_addrg = {
      .bank_base_address = input1_addr,
      .page_size = input1_page_size,
      .data_format = input1_data_format};

  for (uint32_t tile_idx = tile_offset; tile_idx < tile_offset + num_tiles;
       ++tile_idx) {
    cb_reserve_back(input0_cb, 1);
    cb_reserve_back(input1_cb, 1);

    const auto input0_cb_addr = get_write_ptr(input0_cb);
    noc_async_read_tile(tile_idx, input0_addrg, input0_cb_addr);

    const auto input1_cb_addr = get_write_ptr(input1_cb);
    noc_async_read_tile(tile_idx, input1_addrg, input1_cb_addr);

    noc_async_read_barrier();

    cb_push_back(input0_cb, 1);
    cb_push_back(input1_cb, 1);
  }

  DPRINT << "READER END" << ENDL();
}
