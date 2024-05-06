// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <cstdint>


namespace tt {
namespace tt_metal {
namespace ccl {

KernelHandle generate_edm_kernel(
    tt_metal::Program &program,
    Device const* device,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    NOC noc_id);

void generate_edm_kernels_for_ring_or_linear_topology(
    tt_metal::Program &program,
    Device const* device,
    std::vector<ccl::EriscDatamoverBuilder> const& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id,
    // TODO: move to linear/ring topology specific config
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    bool is_linear);

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode);

} // namespace ccl
} // namespace tt_metal
} // namespace tt
