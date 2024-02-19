#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {
struct MyOp {
    bool some_arg;

    // These methods are needed if the operation takes in input tensor and produces output tensors
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

inline Tensor my_op(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    TT_ASSERT(input_tensor_a.shape() == input_tensor_b.shape(), "Input shapes must be the same!");
    auto output = operation::run_without_autoformat(MyOp{false}, {input_tensor_a, input_tensor_b});
    return output.at(0);
}

}  // namespace tt_metal
}  // namespace tt
