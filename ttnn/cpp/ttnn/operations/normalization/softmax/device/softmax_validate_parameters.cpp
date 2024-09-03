// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include "tt_metal/common/assert.hpp"
#include "ttnn/tensor/types.hpp"
#include "softmax_validate_parameters.hpp"

namespace ttnn::operations::normalization {

bool SoftmaxValidateParameters::validateOutput(
    const std::vector<std::string>& output_layouts_str,
    const std::vector<std::string>& output_data_types_str,
    const std::vector<std::vector<uint32_t>>& output_shapes_vectors,
    const std::vector<bool>& output_sharded,
    const std::vector<std::string>& output_orientations_str,
    const std::vector<std::string>& output_memory_layouts,
    const std::vector<std::string>& output_buffer_types,
    OperationProgramConfig& program_config_parameters,
    const std::optional<std::vector<std::vector<uint32_t>>>& output_shard_shapes,
    const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets)
{
    return true;
}

bool SoftmaxValidateParameters::validateInputAndOutput(
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
    const std::optional<std::vector<CoreRangeSet>>& output_core_range_sets)
{
    try
    {
        TT_FATAL((layout_by_index(0, input_layouts_str) == Layout::TILE), "Inputs to softmax must be tilized");
        DataType input_data_type = datatype_by_index(0, input_data_types_str);
        TT_FATAL(input_data_type == DataType::FLOAT32 || input_data_type == DataType::BFLOAT16 || input_data_type == DataType::BFLOAT8_B,
                "Data type must be FLOAT32, BFLOAT16 or BFLOAT8_B");
    }
    catch(const std::runtime_error& e)
    {
        return false;
    }
    return true;
}

}
