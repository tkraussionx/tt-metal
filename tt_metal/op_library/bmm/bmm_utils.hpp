#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

namespace tt {

namespace tt_metal {
    void create_CBs_for_fused_matmul(tt_metal::Program* program, tt_metal::Device* device, tt_xy_pair core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t num_bytes_for_df);
}

}
