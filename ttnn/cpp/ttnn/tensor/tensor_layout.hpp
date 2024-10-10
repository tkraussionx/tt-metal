// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"
#include "enum_types.hpp"

#include <cstddef>
#include <ostream>

namespace tt::tt_metal {

class Size {
public:
    Size(size_t height, size_t width);
    Size(const std::pair<size_t, size_t>& size);
    Size(const std::array<size_t, 2>& size);

    operator std::pair<size_t, size_t>() const;
    operator std::array<size_t, 2>() const;
    operator std::array<uint32_t, 2>() const;

    Size operator/(const Size& rhs) const;
    Size operator*(size_t scalar) const;

    // comparison operator
    bool operator==(const Size& rhs) const;

    size_t height() const;
    size_t width() const;

    // does not have to be a member, but it is easier to find if it is
    Size aligned_to_tile(const Size& tile);

private:
    size_t mHeight = 0;
    size_t mWidth = 0;
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size);

class TensorLayout {
public:
    TensorLayout(DataType dataType, const Size& tileSize, const MemoryConfig& memoryConfig);
    TensorLayout(DataType dataType, Layout layout, const MemoryConfig& memoryConfig);

    Layout get_layout() const { return mLayout; }
    DataType get_data_type() const { return mDataType; }
    const Size& get_tile_size() const { return mTileSize; }
    const MemoryConfig& get_memory_config() const { return mMemoryConfig; }

    std::optional<ShardSpecBuffer> get_shard_spec_buffer(const ttnn::SimpleShape& shape) const;
    size_t get_packed_buffer_size(const ttnn::SimpleShape& shape) const;

    Size get_page_size() const;
    size_t get_page_size_bytes(const ttnn::SimpleShape& shape) const;

    Size get_physical_size(const ttnn::SimpleShape& shape) const;
    ttnn::SimpleShape get_padded_shape(const ttnn::SimpleShape& shape) const;

    Size get_tile_alignment_padding(const ttnn::SimpleShape& shape) const;

private:
    Size get_sharded_page_size() const;

    uint32_t get_page_elements_count(const ttnn::SimpleShape& shape) const;
    uint32_t get_header_size_bytes() const;
    uint32_t element_size_bytes() const;

    Layout mLayout = Layout::ROW_MAJOR;
    DataType mDataType = DataType::BFLOAT16;
    Size mTileSize = {32, 32};
    MemoryConfig mMemoryConfig;
};

} // tt::tt_metal
