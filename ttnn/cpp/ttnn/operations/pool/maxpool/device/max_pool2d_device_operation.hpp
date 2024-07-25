// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

#include "ttnn/experimental/tt_dnn/op_library/sliding_window_op_infra/sliding_window.hpp"

namespace ttnn::operations::pool {

struct MaxPoolNew {
    SlidingWindowConfig sliding_window_config_;
    MemoryConfig out_mem_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::OpPerformanceModel create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "sliding_window_config",
        "out_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->sliding_window_config_),
            std::cref(this->out_mem_config_));
    }
};


} // namespace ttnn::operations::pool
