// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
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

    if (input_shape.size() < 4) {
        const int C = input_shape[0], H = input_shape[1], W = input_shape[2];
        std::cout << "Adding batch dim" << std::endl;
        preprocessed_tensor = ttnn::operations::core::reshape<4>(input_tensor, {1, C, H, W});
        dim += 1; // since we added added a dim to the front
    }

    if (dim != 3) {
        preprocessed_tensor = ttnn::transpose(preprocessed_tensor, dim, 3, mem_config);
    }

    std::vector<Tensor> splits = split_last_dim_two_chunks_tiled(preprocessed_tensor, mem_config);

    auto post_proc = [&dim,
                      &mem_config,
                      &input_shape](const Tensor &split) {
        Tensor res = Tensor(split);

        if (dim != 3) {
            res = ttnn::transpose(res, dim, 3, mem_config);
        }

        if (input_shape.size() < 4) {
            auto s_shape = split.get_shape();
            const int C = s_shape[1], H = s_shape[2], W = s_shape[3];

            auto sans_batch_dim = ttnn::operations::core::reshape<3>(split, {C, H, W});
            res = sans_batch_dim;
        }

        return res;
    };

    std::transform(splits.begin(), splits.end(), splits.begin(), post_proc);

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
