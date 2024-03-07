// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_copy/moreh_copy_op.hpp"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

namespace tt {

namespace operations {

namespace primary {

namespace {}  // namespace

void MorehCopy::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {}

std::vector<Shape> MorehCopy::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehCopy::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, operation::DEFAULT_OUTPUT_MEMORY_CONFIG)};
}

operation::ProgramWithCallbacks MorehCopy::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    auto &output = output_tensors.at(0);
    return moreh_copy_impl(input, output);
}

Tensor moreh_copy(const Tensor &input, const std::optional<const Tensor> output) {
    return operation::run(MorehCopy{}, {input}, {}, {output}).at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
