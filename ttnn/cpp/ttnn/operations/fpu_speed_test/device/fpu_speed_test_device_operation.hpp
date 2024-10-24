#pragma once

#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::fpu_speed_test {

struct FPUSpeedTestDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_tiles;
        bool fp32_dest_acc_en;
    };

    struct tensor_args_t {
        const Tensor &dummy;
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
        return tensor_args.dummy.get_shape();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.dummy;
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        uint32_t num_tiles, bool fp32_dest_acc_en, const Tensor &dummy) {
        return {
            operation_attributes_t{
                .num_tiles = num_tiles,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
            tensor_args_t{
                .dummy = dummy,
            },
        };
    }
};

}  // namespace ttnn::operations::fpu_speed_test

namespace ttnn::prim {

constexpr auto fpu_speed_test = ttnn::
    register_operation<"ttnn::prim::fpu_speed_test", ttnn::operations::fpu_speed_test::FPUSpeedTestDeviceOperation>();

}  // namespace ttnn::prim
