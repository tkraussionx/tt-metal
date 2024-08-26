// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include "impl/program/program.hpp"
#include "ttnn/tensor/types.hpp"
#include "common/core_coord.h"
#include <optional>
#include <variant>

namespace ttnn::operations::normalization {

struct SoftmaxDefaultProgramConfig{
};
struct SoftmaxShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;
};

using SoftmaxProgramConfig = std::variant<
    SoftmaxDefaultProgramConfig,
    SoftmaxShardedMultiCoreProgramConfig
>;

class SoftmaxValidateParameters
{
public:

    SoftmaxValidateParameters(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const SoftmaxProgramConfig& program_config,
        bool is_scale_causal_mask_hw_dims_softmax,
        bool inplace,
        bool is_causal_mask,
        std::optional<int> scale
    );

    SoftmaxValidateParameters(
        const std::vector<std::string>& input_storage_types_str,
        const std::vector<std::string>& input_layouts_str,
        const std::vector<std::string>& input_data_types_str,
        const std::vector<std::vector<uint32_t>>& input_shapes_vectors,
        const std::vector<bool>& input_sharded,
        const std::vector<std::string>& input_orientations_str,
        std::vector<std::string> input_devices,
        std::vector<bool> has_buffers,
        bool is_scale_causal_mask_hw_dims_softmax,
        bool inplace,
        bool is_causal_mask,
        const std::string& program_config,
        std::optional<int> block_w,
        std::optional<int> block_h,
        std::optional<int> subblock_w,
        std::optional<int> subblock_h,
        std::optional<int> grid_size_x,
        std::optional<int> grid_size_y,
        std::optional<int> scale
    );

    void validate();
    bool validate_parameters();

private:
    std::optional<StorageType> str_to_storage_type(const std::string& storage_type_str);
    std::optional<Layout> str_to_layout(const std::string& layout_str);
    std::optional<DataType> str_to_data_type(const std::string& data_type_str);
    tt::tt_metal::Shape vector_to_shape(const std::vector<uint32_t>& shape_vector);
    std::optional<ShardOrientation> str_to_shard_orientation(const std::string& shard_str);

    std::vector<tt::tt_metal::StorageType> input_storage_types;
    std::vector<tt::tt_metal::Layout> input_layouts;
    std::vector<tt::tt_metal::DataType> input_data_types;
    std::vector<tt::tt_metal::Shape> input_shapes;
    std::vector<bool> input_sharded;
    std::vector<tt::tt_metal::ShardOrientation> input_orientations;
    std::vector<std::string> input_devices;
    std::vector<bool> has_buffers;
    bool is_scale_causal_mask_hw_dims_softmax;
    bool inplace;
    bool is_causal_mask;
    SoftmaxProgramConfig program_config;
    std::optional<int> scale;
};

}
