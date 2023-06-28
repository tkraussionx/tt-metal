#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr                     = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_tile_id                  = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t in0_is_dram               = get_compile_time_arg_val(1);
    // READER COMPILE TIME ARGS
    constexpr uint32_t out_num_tensors           = get_compile_time_arg_val(2);
    constexpr uint32_t out_num_tiles_per_tensor  = get_compile_time_arg_val(3);
    constexpr uint32_t out_num_blocks  = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);


    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;
    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
    #if (tile_dtype_is_bfloat16)
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16
    };
    #else
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Bfp8_b
    };
    #endif

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    for (uint32_t out_tensor = 0; out_tensor < out_num_blocks; out_tensor++) {
        cb_reserve_back(cb_id_in0, block_size);
        for (uint32_t i = 0; i < block_size; i++) {
            noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, block_size);
    }
}
