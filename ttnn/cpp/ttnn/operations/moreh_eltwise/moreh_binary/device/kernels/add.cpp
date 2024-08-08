#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
  // TODO. get runtime arguments.
  uint32_t arg = 0;
  const auto input0_cb = get_arg_val<uint32_t>(arg++);
  const auto input1_cb = get_arg_val<uint32_t>(arg++);
  const auto output_cb = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);

  constexpr auto dst0 = 0;
  constexpr auto first = 0;
  constexpr auto onetile = 1;

  binary_op_init_common(input0_cb, input1_cb, output_cb);
  for (uint32_t i = 0; i < num_tiles; i++) {
    tile_regs_acquire();
    cb_wait_front(input0_cb, onetile);
    cb_wait_front(input1_cb, onetile);
    add_tiles_init(input0_cb, input1_cb);
    add_tiles(input0_cb, input1_cb, first, first, dst0);
    cb_pop_front(input0_cb, onetile);
    cb_pop_front(input1_cb, onetile);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(output_cb, onetile);
    pack_tile(dst0, output_cb);
    cb_push_back(output_cb, onetile);

    tile_regs_release();
  }

  UNPACK(DPRINT << "UNPACK END" << ENDL());
  MATH(DPRINT << "MATH END" << ENDL());
  PACK(DPRINT << "PACK END" << ENDL());
}
} // namespace NAMESPACE
