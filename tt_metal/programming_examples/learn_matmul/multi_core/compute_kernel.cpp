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


    mm_init();

    // for(uint32_t i = 0 ; i<Mt; i++)
    // {
    //     for(uint32_t j = 0 ; j<Nt; j++)
    //     {
            acquire_dst(tt::DstMode::Full);
    //         for(uint32_t k = 0; k<Kt; k++)
    //         {
                cb_wait_front(tt::CB::c_in0, onetile);
                cb_wait_front(tt::CB::c_in1, onetile);
                matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);
                cb_pop_front(tt::CB::c_in0, onetile);
                cb_pop_front(tt::CB::c_in1, onetile);
    //         }
            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(0, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);
            release_dst(tt::DstMode::Full);
    //     }
    // }

    }
}
