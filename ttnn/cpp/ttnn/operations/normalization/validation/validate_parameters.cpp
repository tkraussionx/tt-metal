// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <optional>
#include <string>
#include <vector>
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "validate_parameters.hpp"

namespace ttnn::operations
{
std::optional<tt::tt_metal::StorageType> ValidateParameters::str_to_storage_type(const string& storage_type_str)
{
    if (storage_type_str == "OWNED") return StorageType::OWNED;
    if (storage_type_str == "DEVICE") return StorageType::DEVICE;
    if (storage_type_str == "BORROWED") return StorageType::BORROWED;
    if (storage_type_str == "MULTI_DEVICE") return StorageType::MULTI_DEVICE;
    if (storage_type_str == "MULTI_DEVICE_HOST") return StorageType::MULTI_DEVICE_HOST;
    return std::nullopt;
}

std::optional<tt::tt_metal::Layout> ValidateParameters::str_to_layout(const string& layout_str)
{
    if (layout_str == "ROW_MAJOR") return Layout::ROW_MAJOR;
    if (layout_str == "TILE") return Layout::TILE;
    if (layout_str == "INVALID") return Layout::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::TensorMemoryLayout> ValidateParameters::str_to_memory_layout(const string& memory_layout_str)
{
    if (memory_layout_str == "INTERLEAVED") return TensorMemoryLayout::INTERLEAVED;
    if (memory_layout_str == "SINGLE_BANK") return TensorMemoryLayout::SINGLE_BANK;
    if (memory_layout_str == "HEIGHT_SHARDED") return TensorMemoryLayout::HEIGHT_SHARDED;
    if (memory_layout_str == "WIDTH_SHARDED") return TensorMemoryLayout::WIDTH_SHARDED;
    if (memory_layout_str == "BLOCK_SHARDED") return TensorMemoryLayout::BLOCK_SHARDED;
    return std::nullopt;
}

std::optional<tt::tt_metal::DataType> ValidateParameters::str_to_data_type(const string& data_type_str)
{
    if (data_type_str == "BFLOAT16") return DataType::BFLOAT16;
    if (data_type_str == "FLOAT32") return DataType::FLOAT32;
    if (data_type_str == "UINT32") return DataType::UINT32;
    if (data_type_str == "BFLOAT8_B") return DataType::BFLOAT8_B;
    if (data_type_str == "BFLOAT4_B") return DataType::BFLOAT4_B;
    if (data_type_str == "UINT8") return DataType::UINT8;
    if (data_type_str == "UINT16") return DataType::UINT16;
    if (data_type_str == "INT32") return DataType::INT32;
    if (data_type_str == "INVALID") return DataType::INVALID;
    return std::nullopt;
}

std::optional<tt::tt_metal::BufferType> ValidateParameters::str_to_buffer_type(const string& buffer_type_str)
{
    if (buffer_type_str == "DRAM") return BufferType::DRAM;
    if (buffer_type_str == "L1") return BufferType::L1;
    if (buffer_type_str == "SYSTEM_MEMORY") return BufferType::SYSTEM_MEMORY;
    if (buffer_type_str == "L1_SMALL") return BufferType::L1_SMALL;
    if (buffer_type_str == "TRACE") return BufferType::TRACE;
    return std::nullopt;
}

tt::tt_metal::Shape ValidateParameters::vector_to_shape(const std::vector<uint32_t>& shape_vector)
{
    return tt::tt_metal::Shape(shape_vector);
}


std::optional<tt::tt_metal::ShardOrientation> ValidateParameters::str_to_shard_orientation(const string& shard_str)
{
    if (shard_str == "ROW_MAJOR") return ShardOrientation::ROW_MAJOR;
    if (shard_str == "COL_MAJOR") return ShardOrientation::COL_MAJOR;
    return std::nullopt;
}

std::optional<ttnn::Shape> ValidateParameters::vector_to_shard_shape(const std::vector<uint32_t>& shard_shape_vector)
{
    return ttnn::Shape(shard_shape_vector);
}

Layout ValidateParameters::layout_by_index(const int index, const std::vector<std::string>& layouts_str)
{
    return str_to_layout(layouts_str.at(index)).value();
}

DataType ValidateParameters::datatype_by_index(const int index, const std::vector<std::string>& data_types_str)
{
    return str_to_data_type(data_types_str.at(index)).value();
}

tt::tt_metal::Shape ValidateParameters::shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shapes_vectors)
{
    return vector_to_shape(shapes_vectors.at(index));
}

bool ValidateParameters::is_sharded_by_index(const int index, const std::vector<bool>& shareded_vector)
{
    return shareded_vector.at(0);
}

tt::tt_metal::TensorMemoryLayout ValidateParameters::memory_layout_by_index(const int index, const std::vector<std::string>& memory_layouts)
{
    return str_to_memory_layout(memory_layouts.at(index)).value();
}

ShardOrientation ValidateParameters::shard_orientation_by_index(const int index, const std::vector<std::string>& shards_str)
{
    return str_to_shard_orientation(shards_str.at(index)).value();
}

tt::tt_metal::BufferType ValidateParameters::buffer_type_by_index(const int index, const std::vector<std::string>& buffer_types_str)
{
    return str_to_buffer_type(buffer_types_str.at(index)).value();
}

const uint32_t ValidateParameters::volume(tt::tt_metal::Shape& shape)
{
    auto rank = shape.rank();
    auto volume = 1;
    for (auto index = 0; index < rank; index++) {
        volume *= shape.operator[](index);
    }
    return volume;
}

ttnn::Shape ValidateParameters::shard_shape_by_index(const int index, const std::vector<std::vector<uint32_t>>& shard_shapes)
{
    return vector_to_shard_shape(shard_shapes.at(index)).value();
}

CoreRangeSet ValidateParameters::get_core_range_set_by_index(const int index, const std::vector<CoreRangeSet>& core_range_set)
{
    return core_range_set.at(index);
}

}
