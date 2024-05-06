// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "impl/buffers/buffer.hpp"
#include "impl/kernels/data_types.hpp"
#include "tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "eth_l1_address_map.h"
#include "tensor/tensor_impl.hpp"

// Includes that need to be moved to CCL datastructures header
#include <vector>


using namespace tt::constants;

namespace tt {

namespace tt_metal {





/////////////////////////////////////////////////////////////

// TODO(snijjar): move enable_bidirectional to a topology specific config
std::size_t decide_number_of_edm_channels(ccl::CCLOpConfig const& ccl_op_config, std::size_t max_num_workers, bool enable_bidirectional) {
    return ccl_op_config.is_input_sharded() ?
        std::min<uint32_t>(ccl_op_config.get_shard_grid_size(), std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4)) :
        std::min<std::size_t>(max_num_workers, enable_bidirectional ? 8 : 4);
}



// Notes on abbreviations:
// CW = clockwise
// CCW = counter-clockwise
// edm = erisc data mover

// How this reduce_scatter op works:
// For each chip, we have a element range of the input tensor shape that will eventually scatter
// out to it. For all other chunks outside that range, the chip will forward the chunk to the next chip.
// While forwarding the data, the chip will also reduce it with the local input tensor chunk corresponding
// with that received chunk. It will forward the partially reduced chunk.
operation::ProgramWithCallbacks reduce_scatter_with_workers(
    const Tensor& input_tensor,
    Tensor& output_tensor, ReduceOpMath reduce_op,
    const uint32_t reduce_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ccl::Topology topology) {

    /// Constants/Configuration
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;
    auto op_config = ccl::CCLOpConfig(input_tensor, output_tensor);
    bool is_linear = topology == tt::tt_metal::ccl::Topology::Linear;
    auto num_edm_channels = decide_number_of_edm_channels(op_config, 8, false);
    auto const& edm_builder = create_erisc_datamover_builder(num_edm_channels, op_config.get_page_size(), ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode);

    std::vector<ccl::EriscDatamoverBuilder> cw_per_link_edm_builders(num_links, edm_builder);
    std::vector<ccl::EriscDatamoverBuilder> ccw_per_link_edm_builders(num_links, edm_builder);
    //////////////////

    tt_metal::Program program{};
    const auto& device = input_tensor.device();




    TT_ASSERT(false, "Not implemented yet");
}
} // namespace tt_metal

} // namespace tt
