#pragma once

#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::sfpshft2_test {

struct SFPSHFT2TestDeviceOperation {
    struct operation_attributes_t {};

    struct tensor_args_t {
        const Tensor &input;
        const Tensor &output;
    };

    using shape_return_value_t = Shape;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {};
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {}
    };

    using program_factory_t = std::variant<SingleCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return SingleCore{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.output.get_shape();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.output;
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor &input, const Tensor &output) {
        return {
            operation_attributes_t{
            },
            tensor_args_t{
                .input = input,
                .output = output,
            },
        };
    }
};

}  // namespace ttnn::operations::sfpshft2_test

namespace ttnn::prim {

constexpr auto sfpshft2_test = ttnn::
    register_operation<"ttnn::prim::sfpshft2_test", ttnn::operations::sfpshft2_test::SFPSHFT2TestDeviceOperation>();

}  // namespace ttnn::prim
