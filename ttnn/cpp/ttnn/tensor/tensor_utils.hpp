// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {
// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(Tensor conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

// Converts convolution weights to depthwise layout with broadcasted weights
Tensor convert_conv_weight_tensor_to_depthwise_layout(Tensor conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype);

const ttnn::SimpleShape infer_dims_for_reshape(const Tensor& tensor, const std::vector<int32_t>& shape);

// TODO: Remove this once we switch to SimpleShape .volume()
static std::size_t compute_volume(const tt::tt_metal::LegacyShape& shape) {
    size_t volume = 1;
    for (auto index = 0; index < shape.rank(); index++) {
        volume *= shape[index];
    }
    return volume;
}

static std::vector<uint32_t> compute_strides(const ttnn::SimpleShape& shape) {
    if (shape.rank() == 0)
        return {};

    auto num_elements = shape.volume();
    std::vector<uint32_t> strides;
    for (std::int32_t index = 0; index < shape.rank(); index++) {
        if (shape[index] == 0) {
            // Insert 0 to indicate no memory access for this dimension
            strides.push_back(0);
            continue;
        }

        num_elements /= shape[index];
        strides.push_back(num_elements);
    }
    return strides;
}

static int compute_flat_indices(const vector<int>& indices, const vector<std::uint32_t> strides) {
    int flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

static std::size_t compute_buffer_size(const ttnn::SimpleShape& shape, DataType data_type, const std::optional<Tile>& tile = std::nullopt) {
    const size_t volume = shape.volume();
    auto tile_hw = tile.has_value() ? tile->get_tile_hw() : constants::TILE_HW;
    if (data_type == DataType::BFLOAT8_B) {
        auto tile_volume = tile.has_value() ? tile->get_tile_volume(DataFormat::Bfp8_b) : constants::BFLOAT8_B_TILE_HW;
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat8_b_volume = volume / tile_hw * tile_volume;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat8_b_volume / sizeof(std::uint32_t);
    }
    if (data_type == DataType::BFLOAT4_B) {
        auto tile_volume = tile.has_value() ? tile->get_tile_volume(DataFormat::Bfp4_b) : constants::BFLOAT4_B_TILE_HW;
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat4_b_volume = volume / tile_hw * tile_volume;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat4_b_volume / sizeof(std::uint32_t);
    }
    return volume;
}

constexpr auto compute_flat_input_index = [](const auto& indices, const auto& strides) {
    uint32_t flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

bool is_arch_gs(const tt::ARCH& arch);
bool is_arch_whb0(const tt::ARCH& arch);

bool is_cpu_tensor(const Tensor& tensor);
bool is_device_tensor(const Tensor& tensor);

// Given a multi-device tensor, and a function that transforms a tensor, apply the function to all per-device
// tensors.
Tensor transform(const Tensor& tensor, std::function<Tensor(const Tensor&)> transform_func);

// Given a multi-device tensor, and a callable, apply the function to all per-device tensors.
void apply(const Tensor& tensor, std::function<void(const Tensor&)> callable);

// Given a multi-device tensor, return all the devices it is mapped to.
std::vector<Device*> get_devices(const Tensor& multi_device_tensor);

uint32_t num_buffers_in_tensor(const Tensor& tensor);

Tensor get_shard_for_device(
    const Tensor& tensor, Device* target_device, std::optional<int> buffer_index = std::nullopt);

void insert_buffer_and_shape_for_device(
    Device* target_device,
    const Tensor& shard,
    Tensor& tensor_to_modify,
    std::optional<int> buffer_index = std::nullopt);

Tensor copy_borrowed_tensor_in_async_mode(Device* worker, const Tensor& tensor);

template <typename TensorContainer>
auto get_device_tensors(Device* device, const TensorContainer& input_tensors) {
    // Could be Tensor, const Tensor, std::optional<Tensor>, or std::optional<const Tensor>
    using ValueType = typename TensorContainer::value_type;

    // We need a way to extract the underlying Tensor type (const or non-const) from ValueType
    // and to decide whether we are dealing with an optional type.
    using IsOptional = std::conditional_t<
        std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, std::optional<const Tensor>>,
        std::true_type,
        std::false_type>;
    using TensorType = std::conditional_t<
        std::is_same_v<ValueType, std::optional<Tensor>> || std::is_same_v<ValueType, Tensor>,
        Tensor,
        const Tensor>;

    // Result container type adjustment based on input type
    using ResultType = std::conditional_t<IsOptional::value, std::optional<TensorType>, TensorType>;
    std::vector<ResultType> transformed_tensors;

    for (const auto& tensor : input_tensors) {
        if constexpr (IsOptional::value) {
            if (tensor.has_value()) {
                transformed_tensors.emplace_back(get_device_tensor(tensor.value(), device));
            } else {
                transformed_tensors.emplace_back(std::nullopt);
            }
        } else {
            transformed_tensors.emplace_back(get_device_tensor(tensor, device));
        }
    }
    return transformed_tensors;
}

inline bool is_tensor_on_device(const ttnn::Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

inline bool is_tensor_on_multi_device(const ttnn::Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE;
}

inline bool is_tensor_on_device_or_multidevice(const ttnn::Tensor& tensor) {
    return is_tensor_on_device(tensor) or is_tensor_on_multi_device(tensor);
}

template<class T>
inline uint32_t get_batch_size(const T& shape) {
    uint32_t result = 1;
    for (auto i = 0; i < shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

}  // namespace tt_metal

}  // namespace tt
