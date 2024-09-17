// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/run_operation.hpp"
#include "device/split_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/cpp/ttnn/operations/core/core.hpp"
#include <algorithm>
#include <format>

namespace ttnn::operations::data_movement {


namespace detail {

    std::vector<Tensor> impl_split_last_dim_two_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {

        auto input_shape = input_tensor.get_legacy_shape();
        auto padded_input_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_shape);
        ttnn::operations::experimental::auto_format::FormatParams input_format_params = {.pad_shape = padded_input_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
        return operation::run_with_autoformat(SplitDeviceOperation{2, 3, mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE, Layout::TILE});
    }

    std::vector<Tensor> split_last_dim_two_chunks_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {
        const auto shape = input_tensor.get_legacy_shape();

        const bool add_batch_dim_required = shape.size() < 4;
        const bool batch_reshape_required = shape[0] > 1;

        tt::log_info("add_batch_dim_required {}", add_batch_dim_required);

        if (batch_reshape_required) {
            const int W = 1, Z = shape[0] * shape[1], Y = shape[2], X = shape[3];
            const Tensor &reshaped_tensor = ttnn::reshape_on_device(input_tensor, 1, -1, Y, X, mem_config);

            auto part_reshaped = impl_split_last_dim_two_chunks_tiled(reshaped_tensor, mem_config);

            std::vector<Tensor> results;
            results.reserve(part_reshaped.size());
            for (auto &part : part_reshaped) results.emplace_back(ttnn::reshape_on_device(part, -1, shape[1], Y, X / 2, mem_config));

            return results;
        }

        return impl_split_last_dim_two_chunks_tiled(input_tensor, mem_config);
    }


std::vector<Tensor> split_dim_two_chunks_tiled(
    const Tensor &input_tensor, int dim /* = 3 */, const MemoryConfig &mem_config /* = default */) {
    auto input_shape = input_tensor.get_legacy_shape();
    std::vector<Tensor> results;
    results.reserve(2); // two chunks

    Tensor preprocessed_tensor = Tensor(input_tensor);
    const bool needs_batch_dim = input_shape.size() < 4;
    const bool needs_transpose = dim != 3;
    const bool needs_bilateral_padding = input_shape.without_padding()[dim] < tt::constants::TILE_WIDTH;

    if (needs_batch_dim) {
        const int C = input_shape[0], H = input_shape[1], W = input_shape[2];
        std::cout << "Adding batch dim" << std::endl;
        input_shape = {1, C, H, W};
        preprocessed_tensor = input_tensor.reshape(input_shape);
        dim += 1; // since we added added a dim to the front
    }

    if (needs_transpose) {
        preprocessed_tensor = ttnn::transpose(preprocessed_tensor, dim, 3, mem_config);
        std::swap(input_shape[dim], input_shape[3]);
    }

    uint32_t pad_amount;
    if (needs_bilateral_padding) {
        auto unpadded_shape = preprocessed_tensor.get_legacy_shape().without_padding();
        pad_amount = tt::constants::TILE_WIDTH - (unpadded_shape[3] / 2);
        TT_FATAL(unpadded_shape[3] % 2 == 0, "Split dim must be divisble by 2.");
        uint32_t pad_amount = tt::constants::TILE_WIDTH - (unpadded_shape[3] / 2);
        preprocessed_tensor = preprocessed_tensor.to(Layout::ROW_MAJOR);
        preprocessed_tensor = preprocessed_tensor.unpad_from_tile(unpadded_shape);

        uint32_t queue_id = 0;
        std::vector<std::pair<uint32_t, uint32_t>> pad_spec = {{pad_amount, pad_amount}};
        preprocessed_tensor = ttnn::pad(queue_id,
                                        preprocessed_tensor,
                                        pad_spec,
                                        0.0,
                                        true,
                                        std::nullopt); // pad last dim bilaterally by pad amount

        TT_FATAL(preprocessed_tensor.get_shape()[3] / tt::constants::TILE_WIDTH == 2, "Should have two splits here.");
    }

    std::vector<Tensor> splits = split_last_dim_two_chunks_tiled(preprocessed_tensor, mem_config);

    auto post_proc = [&](const Tensor &split, int split_idx) {
        Tensor res = Tensor(split);

        if (needs_bilateral_padding) {
            auto split_shape = split.get_shape();
            const int N = split_shape[0], C = split_shape[1], H = split_shape[2], W = split_shape[3];
            std::vector<uint32_t> split_start = {0,0,0,0};
            std::vector<uint32_t> split_end = {0,0,0,0};

            Tensor rm_split = split.to(Layout::ROW_MAJOR); // convert to row major for unpad
            if (split_idx == 0) {
                // padding on the left
                split_start = {0,0,0,pad_amount};
                split_end = {N, C, H, W+pad_amount-1};
            } else if (split_idx == 1) {
                // padding on the right
                split_start = {0,0,0,0};
                split_end = {N,C,H,W-1};
            } else {
                TT_FATAL(false, "Too many splits!");
            }
            res = rm_split.unpad(tt::tt_metal::Shape(split_start), tt::tt_metal::Shape(split_end));
        }


        if (needs_transpose) {
            res = ttnn::transpose(res, dim, 3, mem_config);
        }

        if (needs_batch_dim) {
            auto s_shape = split.get_shape();
            const int C = s_shape[1], H = s_shape[2], W = s_shape[3];

            auto sans_batch_dim = ttnn::operations::core::reshape<3>(split, {C, H, W});
            res = sans_batch_dim;
        }

        res = res.pad_to_tile(0.0); // ensure we return a tiled tensor.

        return res;
    };

    for (int i = 0; i < splits.size(); i++) {
        results.push_back(post_proc(splits[i], i));
    }

    return splits;
}

}


std::vector<ttnn::Tensor> SplitOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    int64_t& num_splits,
    int64_t& dim,
    const std::optional<MemoryConfig>& memory_config_arg) {

    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    TT_FATAL(num_splits == 2, "Currently only supporting split in 2");
    return detail::split_dim_two_chunks_tiled(input_tensor, dim, memory_config);

}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    int64_t& num_splits,
    int64_t& dim,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, num_splits, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(const ttnn::Tensor& input_tensor, int64_t& num_splits,  int64_t& dim) {
    return invoke(DefaultQueueId, input_tensor, num_splits, dim, std::nullopt);
}

} // ttnn::operations::data_movement namespace
