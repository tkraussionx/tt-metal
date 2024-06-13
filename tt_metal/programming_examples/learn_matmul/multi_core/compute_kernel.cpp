#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
namespace NAMESPACE {
void MAIN {

        constexpr int onetile = 1;
        uint32_t Mt = get_compile_time_arg_val(0);
        uint32_t Kt = get_compile_time_arg_val(1);
        uint32_t Nt = get_compile_time_arg_val(2);
        uint32_t max_per_core_Mt = get_compile_time_arg_val(3);
        uint32_t max_per_core_Nt = get_compile_time_arg_val(4);

        mm_init();

        // int x = src1_addr + 2;
        for(uint32_t m = 0; m<max_per_core_Mt; m++)
        for(uint32_t n = 0; n<max_per_core_Nt; n++)
        {
                acquire_dst(tt::DstMode::Full);
                for(uint32_t k = 0; k<Kt; k++)
                {
                        cb_wait_front(tt::CB::c_in0, onetile);
                        cb_wait_front(tt::CB::c_in1, onetile);
                        matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);
                        cb_pop_front(tt::CB::c_in0, onetile);
                        cb_pop_front(tt::CB::c_in1, onetile);
                }
                cb_reserve_back(tt::CB::c_out0, onetile);
                pack_tile(0, tt::CB::c_out0);
                cb_push_back(tt::CB::c_out0, onetile);
                release_dst(tt::DstMode::Full);
        }

   }
}
