
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
  uint32_t arg = 0;
  const auto output_addr = get_arg_val<uint32_t>(arg++);
  const auto output_cb_id = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);
  const auto tile_offset = get_arg_val<uint32_t>(arg++);

  constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

  const uint32_t output_cb_page_size = get_tile_size(output_cb_id);
  const auto output_cb_data_format = get_dataformat(output_cb_id);
  const InterleavedAddrGenFast<output_is_dram> output_addrg = {
      .bank_base_address = output_addr,
      .page_size = output_cb_page_size,
      .data_format = output_cb_data_format};

  for (uint32_t tile_idx = tile_offset; tile_idx < tile_offset + num_tiles;
       ++tile_idx) {
    cb_wait_front(output_cb_id, 1);
    const auto output_cb_l1_addr = get_read_ptr(output_cb_id);
    noc_async_write_tile(tile_idx, output_addrg, output_cb_l1_addr);
    noc_async_write_barrier();
    cb_pop_front(output_cb_id, 1);
  }

  DPRINT << "Writer END" << ENDL();
}
