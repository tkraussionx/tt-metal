#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
namespace NAMESPACE {
void MAIN {

    constexpr int onetile = 1;
    uint32_t M = get_compile_time_arg_val(0);
    uint32_t K = get_compile_time_arg_val(1);
    uint32_t N = get_compile_time_arg_val(2);

    uint32_t Mt = M/32;
    uint32_t Nt = N/32;
    uint32_t Kt = K/32;

    mm_init();

    for(uint32_t i = 0 ; i<Mt; i++)
    {
        for(uint32_t j = 0 ; j<Nt; j++)
        {
            acquire_dst(tt::DstMode::Full);

            cb_wait_front(tt::CB::c_in0, onetile);
            cb_wait_front(tt::CB::c_in1, onetile);

            matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);
            cb_pop_front(tt::CB::c_in1, onetile);

            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(0, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);

            cb_pop_front(tt::CB::c_in0, onetile);
            release_dst(tt::DstMode::Full);
        }
    }

    }
}
