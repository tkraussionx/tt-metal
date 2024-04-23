// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This is a place holder for when the cpp/ttnn folder structure and ttnn namespace is moved over to tt_eager.
#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {
namespace operations {
namespace transformer {

namespace detail {
inline std::tuple<Tensor, Tensor, Tensor> reshape_outputs_of_split_query_key_value_and_split_heads(
    std::tuple<Tensor, Tensor, Tensor> outputs,
    const uint32_t sequence_size,
    const uint32_t sequence_size_padded,
    const uint32_t sequence_size_kv,
    const uint32_t sequence_size_padded_kv,
    const bool transpose_key) {
    auto &&[query, key, value] = outputs;

    auto batch_size = query.get_shape()[0];
    auto num_heads = query.get_shape()[1];
    auto head_size = query.get_shape()[-1];
    auto head_size_padded = query.get_shape().with_tile_padding()[-1];

    auto num_kv_heads = value.get_shape()[1];

    query = ttnn::reshape(
        query,
        ttnn::Shape(tt::tt_metal::Shape(
            std::array{batch_size, num_heads, sequence_size, head_size},
            std::array{batch_size, num_heads, sequence_size_padded, head_size_padded})));

    if (transpose_key) {
        key = ttnn::reshape(
            key,
            ttnn::Shape(tt::tt_metal::Shape(
                std::array{batch_size, num_kv_heads, head_size, sequence_size_kv},
                std::array{batch_size, num_kv_heads, head_size_padded, sequence_size_padded_kv})));
    } else {
        key = ttnn::reshape(
            key,
            ttnn::Shape(tt::tt_metal::Shape(
                std::array{batch_size, num_kv_heads, sequence_size_kv, head_size},
                std::array{batch_size, num_kv_heads, sequence_size_padded_kv, head_size_padded})));
    }

    value = ttnn::reshape(
        value,
        ttnn::Shape(tt::tt_metal::Shape(
            std::array{batch_size, num_kv_heads, sequence_size_kv, head_size},
            std::array{batch_size, num_kv_heads, sequence_size_padded_kv, head_size_padded})));
    return {query, key, value};
}
}  // namespace detail

inline std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
    const Tensor &input_tensor,
    const std::optional<Tensor> &input_tensor_kv,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_key,
    const std::optional<MemoryConfig> &memory_config) {
    const auto input_shape = input_tensor.get_shape();
    TT_FATAL(input_shape.rank() == 3, "Input Tensor must have strictly 3 dimensions!");
    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::TILE,"Input Tensor must be in a TILE_LAYOUT!");
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE or input_tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE,  "Input_tensor must be on device!");

    const uint32_t sequence_size = input_shape[1];
    const uint32_t sequence_size_padded = input_shape.with_tile_padding()[1];
    uint32_t sequence_size_kv = sequence_size;
    uint32_t sequence_size_kv_padded = sequence_size_padded;

    uint32_t hidden_dim_padded = 0, hidden_dim = 0;
    if (input_tensor_kv.has_value()) {
        const auto input_shape_kv = input_tensor_kv.value().get_shape();
        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[2]/(2*num_kv_heads.value_or(num_heads)) == input_shape[2]/num_heads, "KV tensor hidden size must match Q hidden size");
        hidden_dim = input_shape[2];
        hidden_dim_padded = input_shape.with_tile_padding()[2];
        sequence_size_kv = input_shape_kv[1];
        sequence_size_kv_padded = input_shape_kv.with_tile_padding()[1];
    }
    else {
        hidden_dim = input_shape[2];
        hidden_dim_padded = input_shape.with_tile_padding()[2];
    }

    uint32_t head_size = hidden_dim / num_heads;
    uint32_t padded_head_size = hidden_dim_padded / num_heads;
    TT_FATAL(padded_head_size == head_size, fmt::format("Head size {} cannot have tile padding", head_size));

    if (input_tensor.is_sharded()) {
        const auto input_tensor_4d = input_tensor.reshape(
            input_shape.with_tile_padding()[0],
            1,
            input_shape.with_tile_padding()[1],
            input_shape.with_tile_padding()[2]);

        if (num_kv_heads.value()) {
            auto padded_input_shape_kv = input_tensor_kv.value().get_shape().with_tile_padding();
            auto input_tensor_kv_4d = input_tensor_kv.value().reshape(
                padded_input_shape_kv[0], 1, padded_input_shape_kv[1], padded_input_shape_kv[2]);
            return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            tt::tt_metal::create_qkv_heads_from_separate_tensors(
                input_tensor_4d,
                input_tensor_kv_4d,
                num_heads,
                num_kv_heads.value_or(num_heads),
                transpose_key,
                memory_config.value_or(input_tensor.memory_config())),
            sequence_size,
            sequence_size_padded,
            sequence_size_kv,
            sequence_size_kv_padded,
            transpose_key);
        }
        else {
            return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            tt::tt_metal::create_qkv_heads(
                input_tensor_4d,
                num_heads,
                num_kv_heads.value_or(num_heads),
                transpose_key,
                memory_config.value_or(input_tensor.memory_config())),
            sequence_size,
            sequence_size_padded,
            sequence_size_kv,
            sequence_size_kv_padded,
            transpose_key);
        }
    }
    else {
        const auto input_tensor_4d = input_tensor.reshape(
            input_shape.with_tile_padding()[0],
            1,
            input_shape.with_tile_padding()[1],
            input_shape.with_tile_padding()[2]);
        std::optional<Tensor> input_tensor_kv_4d = std::nullopt;
        if (input_tensor_kv.has_value()) {
            auto padded_input_shape_kv = input_tensor_kv.value().get_shape().with_tile_padding();
            input_tensor_kv_4d = input_tensor_kv.value().reshape(
                padded_input_shape_kv[0], 1, padded_input_shape_kv[1], padded_input_shape_kv[2]);
        }
        const auto outputs = tt::tt_metal::nlp_create_qkv_heads(
            input_tensor_4d,
            input_tensor_kv_4d,
            num_heads,
            num_kv_heads.value_or(num_heads),
            transpose_key,
            memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            {outputs.at(0), outputs.at(1), outputs.at(2)}, sequence_size, sequence_size_padded, sequence_size_kv, sequence_size_kv_padded, transpose_key);
    }
}
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
