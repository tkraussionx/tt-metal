// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt_eager/tensor/tensor.hpp>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/operation_history.hpp"
#include "tt_stl/concepts.hpp"

namespace tt::tt_metal {

namespace operation {

// Temporary changes to allow multi-device ops to work with op profiler
// Should be removed with new profiler + software queue changes
extern bool skip_profile;
extern std::map<chip_id_t, std::reference_wrapper<Program>> skipped_programs;

template<typename ConcreteOperation>
std::vector<Tensor> generic_create_output_tensors(
    const ConcreteOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const DataType output_dtype,
    const Layout output_layout,
    const MemoryConfig& output_mem_config
) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(
            create_device_tensor(output_shape, output_dtype, output_layout, input_tensor.device(), output_mem_config));
    }
    return output_tensors;
}



namespace run_operation_state {
namespace detail {
struct RunOperationState {

    RunOperationState() {}

    void push_composite_parent_name(const char* parent_name) {
        this->composite_parent_names.push_back(parent_name);
    }

    void pop_composite_parent_name() {
        this->composite_parent_names.pop_back();
    }

    bool is_composite_operation() const {
        return not composite_parent_names.empty();
    }

    const auto& get_composite_parent_names() const {
        return this->composite_parent_names;
    }

  private:
    std::vector<const char*> composite_parent_names{};
};

inline RunOperationState OPERATION_STATE{};

}  // namespace detail

inline void push_composite_parent_name(const char* parent_name) {
    detail::OPERATION_STATE.push_composite_parent_name(parent_name);
}

inline void pop_composite_parent_name() {
    detail::OPERATION_STATE.pop_composite_parent_name();
}

inline bool is_composite_operation() {
    return detail::OPERATION_STATE.is_composite_operation();
}

inline const auto& get_composite_parent_names() {
    return detail::OPERATION_STATE.get_composite_parent_names();
}

}  // namespace run_operation_state


namespace detail {
template<typename ReturnType, typename... Args>
struct CompositeOperation {

    const char* name;
    std::function<ReturnType(Args...)> function;

    constexpr ReturnType operator()(Args... args) const {
        run_operation_state::push_composite_parent_name(this->name);
        ReturnType output = this->function(args...);
        run_operation_state::pop_composite_parent_name();
        return output;
    }
};

}  // namespace detail

template<typename ReturnType, typename... Args>
constexpr auto decorate_as_composite(const char* name, std::function<ReturnType(Args...)>&& function) {
  return detail::CompositeOperation<ReturnType, Args...>{.name=name, .function=function};
}

template<typename FunctionType>
constexpr auto decorate_as_composite(const char* name, FunctionType function) {
  return decorate_as_composite(name, std::function(function));
}

#ifdef DEBUG
namespace detail {

template <typename OperationType>
std::string operation_type_to_string() {
    if constexpr (std::is_same_v<OperationType, HostOperation>) {
        return "host";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation>) {
        return "device";
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return "external";
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

static operation_history::TensorRecord create_tensor_record(const Tensor& tensor) {
    return std::visit(
        [&](const auto& storage) -> operation_history::TensorRecord {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), std::nullopt
                };
            }
            else if constexpr (std::is_same_v<T, DeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout(), tensor.memory_config()};
            }
            else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            }
            else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                return operation_history::TensorRecord{
                    tensor.storage_type(), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout()
                };
            } else {
                raise_unsupported_storage<T>();
            }
        },
        tensor.get_storage());
}

template<typename OperationType>
static void append_operation_to_operation_history(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.emplace_back(create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.emplace_back(create_tensor_record(tensor.value()));
        }
    }
    operation_history::append(
        operation_history::OperationRecord{
            operation.get_type_name(),
            operation.attributes(),
            input_tensor_records,
            run_operation_state::get_composite_parent_names()
        }
    );
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {}) {
    tt::log_debug(
        tt::LogOp,
        "Launching Operation: \"{}\" ({})",
        operation.get_type_name(),
        detail::operation_type_to_string<OperationType>());

    if (run_operation_state::is_composite_operation()) {
        tt::log_debug(tt::LogOp, "Composite Parents: {}", run_operation_state::get_composite_parent_names());
    }

    if (not operation.attributes().empty()) {
        tt::log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : operation.attributes()) {
            tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    tt::log_debug(tt::LogOp, "Input Tensors:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors:");
        for (auto index = 0; index < optional_input_tensors.size(); index++) {
            const auto& tensor = optional_input_tensors[index];
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
        }
    }

    tt::log_debug(tt::LogOp, "");

    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(operation, input_tensors, optional_input_tensors);
    }
}
#else

template <typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {}) {}
#endif

std::vector<Tensor> run(
    const HostOperation& operation,
    const std::vector<Tensor>& input_tensors
);

std::vector<Tensor> run(
    CommandQueue& queue,
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {});

std::vector<Tensor> run(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {});

template<typename ConcreteOperation>
inline std::vector<Tensor> run(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {}
) {
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto operation = HostOperation(concrete_op);
        return run(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run(operation, input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

std::vector<Tensor> run_without_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {}
);
template <typename ConcreteOperation>
inline std::vector<Tensor> run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {}) {
    const auto operation = DeviceOperation(concrete_op);
    return run_without_autoformat(operation, input_tensors, optional_input_tensors, optional_output_tensors);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_with_autoformat(operation, input_tensors, optional_input_tensors, pad_value, pad_c);
}

std::vector<Tensor> run_with_autoformat(
    const DeviceOperation& operation,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
);
template<typename ConcreteOperation>
inline std::vector<Tensor> run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {}
) {
    const auto operation = DeviceOperation(concrete_op);
    return run_with_autoformat(operation, input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting);
}

inline void launch_op(
    std::function<Tensor(const std::vector<Tensor>&, const std::vector<std::optional<const Tensor>>&)> op_func,
    const std::vector<Tensor> input_tensors,
    Tensor output_tensor,
    const std::vector<std::optional<const Tensor>> optional_input_tensors = {}
) {
    output_tensor.tensor_attributes->storage = MultiDeviceStorage();
    // Send host side op compile and run to the worker queue
    // Async mode changes for tensor-parallel execution not mainlined. Use the first worker thread.
    for (auto worker : output_tensor.workers) {
        worker->push_work([op_func, input_tensors, optional_input_tensors, output_tensor, worker] () mutable {
            std::visit([=] (auto&& storage) mutable {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    std::vector<Tensor> input_shards = {};
                    std::vector<std::optional<const Tensor>> optional_input_shards = {};

                    for (const auto& input : input_tensors) {
                        auto input_storage = std::get<tt::tt_metal::MultiDeviceStorage>(input.get_storage());
                        auto shard_shape = input_storage.get_tensor_shape_for_device(worker);
                        auto shard_buffer = input_storage.get_buffer_for_device(worker);
                        input_shards.push_back(Tensor{DeviceStorage{shard_buffer}, shard_shape, input.get_dtype(), input.get_layout()});
                    }
                    for (auto& input : optional_input_tensors) {
                        if (input.has_value()) {
                            auto input_storage = std::get<tt::tt_metal::MultiDeviceStorage>(input.value().get_storage());
                            auto shard_shape = input_storage.get_tensor_shape_for_device(worker);
                            auto shard_buffer = input_storage.get_buffer_for_device(worker);
                            optional_input_shards.push_back(Tensor{DeviceStorage{shard_buffer}, shard_shape, input.value().get_dtype(), input.value().get_layout()});
                        }
                        else {
                            optional_input_shards.push_back(std::nullopt);
                        }
                    }
                    auto local_tensor = op_func(input_shards, optional_input_shards);
                    storage.insert_buffer_for_device(worker, std::get<DeviceStorage>(local_tensor.get_storage()).buffer);
                    storage.insert_tensor_shape_for_device(worker, local_tensor.get_legacy_shape());

                    if (not (worker->id())) {
                        output_tensor.set_shape(local_tensor.get_shape()); // Need all gather here??
                        output_tensor.set_dtype(local_tensor.get_dtype());
                        output_tensor.set_layout(local_tensor.get_layout());
                        output_tensor.tensor_attributes->metadata_populated = true;
                    }
                }
            }, output_tensor.tensor_attributes->storage);
        });
    }
}
} //namespace operation

} //namespace tt::tt_metal
