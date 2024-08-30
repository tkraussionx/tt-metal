// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <openssl/md5.h>
#include "hash_id_maker.hpp"

std::string HashIdMaker::base64_encode(const std::string& in) {
    static const std::string base64_chars =
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "0123456789+/";
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

std::string HashIdMaker::to_hash(const std::string& input) {
    // Create a buffer to hold the MD5 digest
    unsigned char digest[MD5_DIGEST_LENGTH];

    // Compute the MD5 hash
    MD5((unsigned char*)input.c_str(), input.length(), digest);

    // Convert the digest to a hexadecimal string
    std::ostringstream md5str;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5str << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }

    return md5str.str();
}

std::string HashIdMaker::create_hex_string(
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
)
{
    std::ostringstream concatenated_string;

    concatenated_string << "batch_sizes(" + std::to_string(batch_size) + ",)";
    concatenated_string << "num_inputs" + std::to_string(num_inputs);
    concatenated_string << "input_a_height" << std::to_string(input_a_height);
    concatenated_string << "input_a_width" << std::to_string(input_a_width);
    concatenated_string << "input_a_dtype" << input_a_dtype;
    concatenated_string << "input_a_layout" << input_a_layout;
    concatenated_string << "input_a_memory_config" << input_a_memory_config;
    concatenated_string << "input_a_sharding_strategy" << input_a_sharding_strategy;
    concatenated_string << "multi_core_program_config" << multi_core_program_config;
    concatenated_string << "is_scale_causal_mask_hw_dims_softmax" << (is_scale_causal_mask_hw_dims_softmax ? "True" : "False");
    concatenated_string << "is_inplace" << (is_inplace ? "True" : "False");
    concatenated_string << "is_causal_mask" << (is_causal_mask ? "True" : "False");
    concatenated_string << "input_a_shard_orientation" << input_a_shard_orientation;
    concatenated_string << "input_b_memory_config" << input_b_memory_config;
    concatenated_string << "softmax_type" << softmax_type;
    concatenated_string << "sweep_namesoftmax";
    std::cout << "concatenated_string=" << concatenated_string.str() << std::endl;
    return to_hash(concatenated_string.str());
}

/* Expects dataype as, for example: FLOAT32 */
std::string HashIdMaker::datatype_from_string(const std::string& datatype_str)
{
    return "DataType." + datatype_str;
}

/* Expects layout as, for example: TILE*/
std::string HashIdMaker::layout_from_string(const std::string& layout_str)
{
    return "Layout." + layout_str;
}

/* Expects memory layout as <INTERLEAVED, SHARDED>, buffer type as <L1, DRAM>, currently shard_spec not supported*/
std::string HashIdMaker::memory_config_from_strings(const std::string& memory_layout_str, const std::string& buffer_str)
{
    return "MemoryConfig(memory_layout=TensorMemoryLayout::"+ memory_layout_str +",buffer_type=BufferType::"+ buffer_str +",shard_spec=std::nullopt)";
}

/* Expects shard scpec as <HEIGHT, WIDTH> */
std::string HashIdMaker::sharding_strategy_from_string(const std::string& shard_strategy_str)
{
    return "ShardStrategy." + shard_strategy_str;
}

/* Expects shard orientation as <COL_MAJOR, ROW_MAJOR> */
std::string HashIdMaker::shard_orientation_from_string(const std::string& shard_orientation_str)
{
    return "ShardOrientation." + shard_orientation_str;
}

std::string HashIdMaker::create_string_for_softmax(
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
    const std::string& buffer_str_2)
{
    std::string input_a_dtype = datatype_from_string(datatype_str);
    std::string input_a_layout = layout_from_string(layout_str);
    std::string input_a_memory_config = memory_config_from_strings(memory_layout_str_1, buffer_str_1);
    std::string input_a_sharding_strategy = sharding_strategy_from_string(shard_strategy_str);
    std::string multi_core_program_config = "<class 'ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig'>";
    std::string input_a_shard_orientation = shard_orientation_from_string(shard_orientation_str);
    std::string input_b_memory_config = memory_config_from_strings(memory_layout_str_2, buffer_str_2);
    std::string softmax_type = "softmax";

    return create_hex_string(
        batch_size,
        num_inputs,
        input_a_height,
        input_a_width,
        input_a_dtype,
        input_a_layout,
        input_a_memory_config,
        input_a_sharding_strategy,
        multi_core_program_config,
        is_scale_causal_mask_hw_dims_softmax,
        is_inplace,+
        is_causal_mask,
        input_a_shard_orientation,
        input_b_memory_config,
        softmax_type
    );
}
