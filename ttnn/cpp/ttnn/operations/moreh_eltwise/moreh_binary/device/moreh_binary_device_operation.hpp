// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::moreh_eltwise {

struct MorehBinaryDeviceOperation {
  struct operation_attributes_t {
    const float scalar0;
    const float scalar1;
    const uint8_t program_selector;
  };

  struct tensor_args_t {
    const Tensor &input_tensor0;
    const Tensor &input_tensor1;
    const std::optional<Tensor> &output_tensor;
  };

  static tt::stl::hash::hash_t
  compute_program_hash(const operation_attributes_t &, const tensor_args_t &);

  static void validate_on_program_cache_miss(const operation_attributes_t &,
                                             const tensor_args_t &);
  static void validate_on_program_cache_hit(const operation_attributes_t &,
                                            const tensor_args_t &);

  using shape_return_value_t = ttnn::Shape;
  static shape_return_value_t
  compute_output_shapes(const operation_attributes_t &, const tensor_args_t &);
  using tensor_return_value_t = Tensor;
  static tensor_return_value_t
  create_output_tensors(const operation_attributes_t &, const tensor_args_t &);

  struct Add {
    struct shared_variables_t {
      KernelHandle reader_kernel_id;
      KernelHandle writer_kernel_id;
      KernelHandle compute_kernel_id;
      CoreCoord grid;
    };
    using cached_program_t =
        ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t
    create(const operation_attributes_t &operation_attributes,
           const tensor_args_t &tensor_args,
           tensor_return_value_t &tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t &cached_program,
        const operation_attributes_t &operation_attributes,
        const tensor_args_t &tensor_args,
        tensor_return_value_t &tensor_return_value);
  };

  struct Fusion {
    struct shared_variables_t {
      KernelHandle reader_kernel_id;
      KernelHandle writer_kernel_id;
      KernelHandle compute_kernel_id;
      CoreCoord grid;
    };
    using cached_program_t =
        ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t
    create(const operation_attributes_t &operation_attributes,
           const tensor_args_t &tensor_args,
           tensor_return_value_t &tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t &cached_program,
        const operation_attributes_t &operation_attributes,
        const tensor_args_t &tensor_args,
        tensor_return_value_t &tensor_return_value);
  };

  using program_factory_t = std::variant<Add, Fusion>;

  static program_factory_t
  select_program_factory(const operation_attributes_t &, const tensor_args_t &);
};

} // namespace ttnn::operations::moreh_eltwise
