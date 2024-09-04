// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include "ttnn/cpp/ttnn/operations/normalization/validation/validate_parameters.hpp"

namespace ttnn::operations::binary {

class BinaryValidateParameters : public ttnn::operations::ValidateParameters
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
    );
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
    );

};

}
