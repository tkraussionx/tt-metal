// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

class HashIdMaker
{
public:

std::string base64_encode(const std::string& in);
std::string to_hash(const std::string& input);
std::string create_hex_string(
    int batch_size,
    int num_inputs,
    int input_a_height,
    int input_a_width,
    const std::string& input_a_dtype,
    const std::string& input_a_layout,
    const std::string& input_a_memory_config,
    const std::string& input_a_sharding_strategy,
    const std::string& multi_core_program_config,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool is_inplace,
    bool is_causal_mask,
    const std::string& input_a_shard_orientation,
    const std::string& input_b_memory_config,
    const std::string& softmax_type
);
std::string datatype_from_string(const std::string& datatype_str);
std::string layout_from_string(const std::string& layout_str);
std::string memory_config_from_strings(const std::string& memory_layout_str, const std::string& buffer_str);
std::string sharding_strategy_from_string(const std::string& shard_strategy_str);
std::string shard_orientation_from_string(const std::string& shard_orientation_str);
std::string create_string_for_softmax(
    int batch_size,
    int num_inputs,
    int input_a_height,
    int input_a_width,
    const std::string& datatype_str,
    const std::string& layout_str,
    const std::string& memory_layout_str_1,
    const std::string& buffer_str_1,
    const std::string& shard_strategy_str,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool is_inplace,
    bool is_causal_mask,
    const std::string& shard_orientation_str,
    const std::string& memory_layout_str_2,
    const std::string& buffer_str_2);

};
