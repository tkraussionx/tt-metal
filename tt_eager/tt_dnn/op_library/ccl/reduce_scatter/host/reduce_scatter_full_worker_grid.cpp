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

// TODO: move to CCL datastructures header
struct ccl::CCLOpConfig {
   public:
    CCLOpConfig(const Tensor& input_tensor, const Tensor &output_tensor) :
        is_input_sharded(input_tensor.is_sharded()),
        is_output_sharded(output_tensor.is_sharded()),
        page_size(input_tensor.buffer()->page_size()),
        input_shard_size_bytes(
            input_tensor.is_sharded() ?
                (input_tensor.buffer()->page_size() * input_tensor.buffer()->shard_spec().tensor2d_shape[0] * input_tensor.buffer()->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores() :
                std::nullopt),
        output_shard_size_bytes(
            output_tensor.is_sharded() ?
                (output_tensor.buffer()->page_size() * output_tensor.buffer()->shard_spec().tensor2d_shape[0] * output_tensor.buffer()->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores() :
                std::nullopt),
        edm_semaphore_l1_base_address(),
        edm_buffers_l1_base_address()
    {
        TT_ASSERT(!is_input_sharded || input_shard_size_bytes.has_value());
        TT_ASSERT(!is_output_sharded || output_shard_size_bytes.has_value());
    }

    uint32_t get_input_shard_size_bytes() const {
        TT_ASSERT(input_shard_size_bytes.has_value());
        return input_shard_size_bytes.value();
    }
    uint32_t get_output_shard_size_bytes() const {
        TT_ASSERT(output_shard_size_bytes.has_value());
        return output_shard_size_bytes.value();
    }
    uint32_t get_page_size() const {
        return page_size;
    }
    bool is_input_sharded() const {
        return is_input_sharded;
    }
    bool is_output_sharded() const {
        return is_output_sharded;
    }

   private:
    std::optional<uint32_t> input_shard_size_bytes; // TODO: split off into CCL op input config ()
    std::optional<uint32_t> output_shard_size_bytes; // TODO: split off into CCL op input config ()
    uint32_t edm_semaphore_l1_base_address;
    uint32_t edm_buffers_l1_base_address;
    uint32_t page_size;
    bool is_input_sharded;
    bool is_output_sharded;
};

struct EriscDatamoverConfig {
    static constexpr std::size_t total_l1_buffer_space = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    static constexpr std::size_t usable_l1_base_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    static constexpr std::size_t semaphore_size = 4;
    static constexpr std::size_t handshake_location_size = 16; // ethernet word size
    static constexpr std::size_t eth_word_size_bytes = 16;

    static uint32_t get_edm_handshake_address() {
        return usable_l1_base_address;
    }
    static uint32_t get_semaphores_base_address(std::size_t num_edm_channels) {
        return usable_l1_base_address + handshake_location_size;
    }
    static uint32_t get_buffers_base_address(std::size_t num_edm_channels) {
        uint32_t base_address = round_up(get_semaphores_base_address(num_edm_channels) + num_edm_channels * semaphore_size, eth_word_size_bytes);
        TT_ASSERT(base_address % eth_word_size_bytes == 0);
        return base_address;
    }
    static uint32_t compute_buffer_size(std::size_t num_edm_channels, uint32_t page_size = eth_word_size_bytes) {
        page_size = std::max<uint32_t>(page_size, eth_word_size_bytes);
        uint32_t buffer_size = round_down((total_l1_buffer_space - get_buffers_base_address(num_edm_channels)) / (num_edm_channels), page_size);
        TT_ASSERT(buffer_size > 0 && buffer_size % page_size == 0);
        return buffer_size;
    }
};


ccl::EriscDatamoverBuilder create_erisc_datamover_builder(std::size_t num_channels, uint32_t page_size, ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode) {

    std::vector<uint32_t> edm_sem_addresses(num_channels, 0);
    std::vector<uint32_t> edm_buffer_addresses(num_channels, 0);

    uint32_t edm_sem_addr = EriscDatamoverConfig::get_semaphores_base_address(num_channels);
    uint32_t edm_buffer_addr = EriscDatamoverConfig::get_buffers_base_address(num_channels);
    const uint32_t buffer_size = EriscDatamoverConfig::compute_buffer_size(num_channels, page_size);
    for (std::size_t c = 0; c < num_channels; ++c) {
        edm_sem_addresses.push_back(edm_sem_addr);
        edm_sem_addr += EriscDatamoverConfig::semaphore_size;
        edm_buffer_addresses.push_back(edm_buffer_addr);
        edm_buffer_addr += buffer_size;
        TT_ASSERT((c == 0) || (edm_buffer_addresses.back() != edm_buffer_addresses.front()));
        TT_ASSERT((c == 0) || (edm_sem_addresses.back() != edm_sem_addresses.front()));
    }

    return ccl::EriscDatamoverBuilder(
        buffer_size, EriscDatamoverConfig::get_edm_handshake_address(), edm_sem_addresses, edm_buffer_addresses, buffer_sharing_mode);
}


/////////////////////////////////////////////////////////////

// TODO(snijjar): move enable_bidirectional to a topology specific config
std::size_t decide_number_of_edm_channels(ccl::CCLOpConfig const& ccl_op_config, std::size_t num_workers, bool enable_bidirectional) {
    return std::min<std::size_t>(num_workers, enable_bidirectional ? 8 : 4);
}

KernelHandle create_worker_kernel...

KernelHandle generate_edm_kernel(
    tt_metal::Program &program,
    Device const* device,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    NOC noc_id) {
    log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
    edm_builder.dump_to_log();

    // auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);

    std::vector<uint32_t> const& edm_clockwise_kernel_rt_args = edm_builder.emit_runtime_args();
    // Ethernet Kernels
    std::vector<uint32_t> eth_sender_ct_args = edm_builder.emit_compile_time_args();

    auto eth_sender_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
        eth_core,
        tt_metal::EthernetConfig{.noc=noc_id, .compile_args=eth_sender_ct_args});


    tt_metal::SetRuntimeArgs(
        program,
        eth_sender_kernel,
        eth_core,
        edm_clockwise_kernel_rt_args);

    // eth_sender_kernels.push_back(eth_sender_kernel);
    // log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_core.x, eth_sender_core.y);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (auto const& s : edm_clockwise_kernel_rt_args) {
        ss << "\t" << s << "\n";
    }
    log_trace(tt::LogOp, "{}", ss.str());

    return eth_sender_kernel;
}

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
    bool is_linear) {

    auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
    auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }
    for (uint32_t i = 0; i < num_links; ++i) {
        bool is_clockwise_direction_edm_enabled = !is_linear || ring_index != ring_size - 1;
        if (is_clockwise_direction_edm_enabled) {
            auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);
            log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
            auto eth_sender_kernel = generate_edm_kernel(
                program,
                device,
                clockwise_edm_builders.at(i),
                eth_sender_core,
                sender_noc);
            // eth_sender_kernels.push_back(eth_sender_kernel);
            log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_core.x, eth_sender_core.y);
        }

        bool is_counter_clockwise_direction_edm_enabled = !is_linear || ring_index != 0;
        if (is_counter_clockwise_direction_edm_enabled) {
            log_trace(tt::LogOp, "EDM COUNTER CLOCKWISE KERNEL RT ARGS: ");
            auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx);
            auto eth_receiver_kernel = generate_edm_kernel(
                program,
                device,
                counter_clockwise_edm_builders.at(i),
                eth_receiver_core,
                receiver_noc);
            log_trace(tt::LogOp, "RingIndex: {}. Link {}. Counter-clockwise EDM Core (x={},y={})", ring_index, i, eth_receiver_core.x, eth_receiver_core.y);
        }

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

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
    bool is_linear = topology == all_gather_op::Topology::Linear;
    auto num_edm_channels = decide_number_of_edm_channels(op_config, num_workers, false);
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
