// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <optional>
#include <variant>
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"


namespace ttnn::operations::primary {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single
// core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<unary::UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<unary::UnaryWithParam> fused_activation;
    bool mcast_in0;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<unary::UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {};

struct MatmulMultiCoreNonOptimizedReuseProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreNonOptimizedReuseProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>;

class MatmulValidateParameters
{
public:

    MatmulValidateParameters();

    void validate();
    bool validate_parameters();

private:
    MatmulProgramConfig create_matmul_program_config();
    std::optional<StorageType> str_to_storage_type(const std::string& storage_type_str);
    std::optional<Layout> str_to_layout(const std::string& layout_str);
    std::optional<DataType> str_to_data_type(const std::string& data_type_str);
    tt::tt_metal::Shape vector_to_shape(const std::vector<uint32_t>& shape_vector);
    std::optional<ShardOrientation> str_to_shard_orientation(const std::string& shard_str);


    std::vector<tt::tt_metal::StorageType> input_storage_types;
    std::vector<tt::tt_metal::Layout> input_layouts;
    std::vector<tt::tt_metal::TensorMemoryLayout> input_memory_layouts;
    std::vector<tt::tt_metal::Shape> input_shapes;
    std::vector<tt::tt_metal::Shape> input_shapes_with_padding;
    std::vector<tt::tt_metal::Shape> input_shard_shapes;
    std::vector<tt::tt_metal::DataType> input_data_types;
    std::vector<bool> has_buffers;
    std::vector<std::string> input_devices;
    std::vector<bool> input_sharded;
    std::vector<tt::tt_metal::BufferType> input_buffer_types;
    std::vector<tt::tt_metal::ShardOrientation> input_orientations;
    std::vector<uint32_t> input_num_cores;
    std::vector<std::string> input_grids;
    std::vector<tt::umd::xy_pair> input_start_coordinates;
    std::vector<tt::umd::xy_pair> input_end_coordinates;
    const std::optional<bool> bcast_batch = std::nullopt;
    bool untilize_out;
    tt::tt_metal::DataType output_data_type;
    tt::tt_metal::TensorMemoryLayout output_layout;
    tt::tt_metal::BufferType output_buffer_type;
    bool output_sharded;
    bool has_bias;
    DeviceComputeKernelConfig compute_kernel_config;
};

}
