// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <string>
#include <vector>
#include <variant>
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_types.hpp"
#include "ttnn/operations/matmul/device/matmul_types.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/core_coord.h"

namespace ttnn::operations
{

using OperationProgramConfig = std::variant<
    normalization::SoftmaxProgramConfig,
    matmul::MatmulProgramConfig
>;

class ValidateParameters
{
public:
    virtual bool validateOutput(
        const std::vector<std::string>& output_layouts_str,
        const std::vector<std::string>& output_data_types_str,
        const std::vector<std::vector<uint32_t>>& output_shapes_vectors,
        const std::vector<bool>& output_sharded,
        const std::vector<std::string>& output_orientations_str,
        const std::vector<std::string>& output_memory_layouts,
        const std::vector<std::string>& output_buffer_types,
        OperationProgramConfig& program_config_parameters,
        const std::optional<std::vector<std::vector<uint32_t>>>& output_shard_shapes,
        const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets
    ) = 0;
    virtual bool validateInputAndOutput(
        const std::vector<std::string>& input_layouts_str,
        const std::vector<std::string>& input_data_types_str,
        const std::vector<std::vector<uint32_t>>& input_shapes_vectors,
        const std::vector<bool>& input_sharded,
        const std::vector<std::string>& input_orientations_str,
        const std::vector<std::string>& input_memory_layouts,
        const std::vector<std::string>& input_buffer_types,
        const std::vector<std::string>& output_layouts_str,
        const std::vector<std::string>& ouput_data_types_str,
        const std::vector<std::vector<uint32_t>>& output_shapes_vectors,
        const std::vector<bool>& output_sharded,
        const std::vector<std::string>& output_orientations_str,
        const std::vector<std::string>& output_memory_layouts,
        const std::vector<std::string>& output_buffer_types,
        OperationProgramConfig& program_config_parameters,
        const std::optional<std::vector<std::vector<uint32_t>>>& input_shard_shapes,
        const std::optional<std::vector<std::vector<uint32_t>>>& output_shard_shapes,
        const std::optional<std::vector<CoreRangeSet>>& input_core_range_sets,
        const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets
    ) = 0;

protected:
    std::optional<StorageType> str_to_storage_type(const std::string& storage_type_str);
    std::optional<Layout> str_to_layout(const std::string& layout_str);
    std::optional<DataType> str_to_data_type(const std::string& data_type_str);
    tt::tt_metal::Shape vector_to_shape(const std::vector<uint32_t>& shape_vector);
    std::optional<ShardOrientation> str_to_shard_orientation(const std::string& shard_str);
    std::optional<tt::tt_metal::TensorMemoryLayout> str_to_memory_layout(const string& memory_layout_str);
    std::optional<tt::tt_metal::BufferType> str_to_buffer_type(const string& buffer_type_str);
    std::optional<ttnn::Shape> vector_to_shard_shape(const std::vector<uint32_t>& shard_shape_vector);
    Layout layout_by_index(const int index, const std::vector<std::string>& layouts_str);
    DataType datatype_by_index(const int index, const std::vector<std::string>& data_types_str);
    tt::tt_metal::Shape shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shapes_vectors);
    bool is_sharded_by_index(const int index, const std::vector<bool>& sharded_vector);
    tt::tt_metal::TensorMemoryLayout memory_layout_by_index(const int index, const std::vector<std::string>& memory_layouts);
    tt::tt_metal::ShardOrientation shard_orientation_by_index(const int index, const std::vector<std::string>& shards_str);
    tt::tt_metal::BufferType buffer_type_by_index(const int index, const std::vector<std::string>& buffer_types_str);
    const uint32_t volume(tt::tt_metal::Shape& shape);
    ttnn::Shape shard_shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shard_shapes);
    CoreRangeSet get_core_range_set_by_index(const int index, const std::vector<CoreRangeSet>& core_range_set);
};

}
