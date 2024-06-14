// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "tt_eager/tensor/types.hpp"
#include "tt_metal/common/math.hpp"

namespace ttnn {
struct Dim2d {
    std::uint32_t y = 0;
    std::uint32_t x = 0;

    constexpr Dim2d() = default;
    constexpr Dim2d(std::uint32_t y, std::uint32_t x) : y(y), x(x) {}
    constexpr Dim2d(CoreCoord coord) : y(coord.y), x(coord.x) {}
    constexpr bool operator==(Dim2d rhs) const { return y == rhs.y && x == rhs.x; }
    constexpr Dim2d align(Dim2d align) const { return Dim2d(::tt::round_up(y, align.y), ::tt::round_up(x, align.x)); }
    constexpr std::uint32_t volume() const { return y * x; }
};

struct ShardShape {
    Dim2d dims;
    constexpr ShardShape() = default;
    constexpr ShardShape(std::uint32_t y, std::uint32_t x) : dims(y, x) {}
    constexpr bool operator==(ShardShape rhs) const { return dims == rhs.dims; }
};

struct GridShape {
    Dim2d dims;
    constexpr GridShape() = default;
    constexpr GridShape(std::uint32_t y, std::uint32_t x) : dims(y, x) {}
    constexpr bool operator==(GridShape rhs) const { return dims == rhs.dims; }
};

struct TileShape {
    Dim2d dims;
    constexpr TileShape() = default;
    constexpr TileShape(std::uint32_t y, std::uint32_t x) : dims(y, x) {}
    constexpr TileShape(Dim2d dims) : dims(dims) {}
    constexpr bool operator==(TileShape rhs) const { return dims == rhs.dims; }
};

struct Layout2 {
    enum class OOBVal { Any, Zero, Infinity, NegativeInfinity };

    ttnn::Shape shape;
    std::vector<std::uint32_t> strides;
    ShardShape shard_shape;
    TileShape tile_shape;
    OOBVal oob_val = OOBVal::Any;

    Layout2() : shape({0xFF, 0xFF, 0xFF, 0xFF}) {}
    Layout2(
        Shape shape,
        ShardShape shard_shape,
        TileShape tile_shape,
        std::vector<std::uint32_t> strides = {},
        OOBVal oob_val = OOBVal::Any) :
        shape(shape),
        strides(strides.empty() ? calculate_strides(shape, default_align(shape.size(), shard_shape.dims)) : strides),
        shard_shape(shard_shape),
        tile_shape(tile_shape) {
        validate();
    }

    Layout2(
        Shape shape,
        GridShape grid_shape,
        TileShape tile_shape,
        std::vector<std::uint32_t> strides = {},
        OOBVal oob_val = OOBVal::Any) :
        shape(shape),
        strides(strides.empty() ? calculate_strides(shape, default_align(shape.size(), grid_shape.dims)) : strides),
        tile_shape(tile_shape) {
        Dim2d collapsed_shape = get_collapsed_shape2d();
        shard_shape = ShardShape(collapsed_shape.y / grid_shape.dims.y, collapsed_shape.x / grid_shape.dims.x);
        TT_ASSERT(grid_shape == get_grid_shape());
        validate();
    }

    static std::vector<std::uint32_t> default_align(std::uint32_t rank, Dim2d dims) {
        std::vector<std::uint32_t> align(rank, 1);
        align[0] = dims.y;
        align[rank - 1] = dims.x;
        return align;
    }

    static std::vector<std::uint32_t> calculate_strides(const ttnn::Shape &shape, std::vector<std::uint32_t> align) {
        TT_ASSERT(shape.size() == align.size());
        std::vector<std::uint32_t> strides(shape.size() + 1);
        strides[shape.size()] = 1;
        for (std::int32_t i = (std::int32_t)shape.size() - 1; i >= 0; --i) {
            strides[i] = ::tt::round_up(shape[i] * strides[i + 1], align[i]);
        }
        return strides;
    }

    static std::vector<std::uint32_t> replace_stride(
        std::vector<std::uint32_t> strides, int dim, std::uint32_t new_stride) {
        if (dim < 0) {
            dim += strides.size();
        }
        TT_ASSERT(dim != (int)(strides.size() - 1), "Cannot replace the last stride");
        TT_ASSERT(dim < (int)strides.size());
        std::uint32_t old_stride = strides[dim];
        for (std::int32_t i = 0; i <= dim; ++i) {
            TT_ASSERT(strides[i] % old_stride == 0);
            strides[i] /= old_stride;
            strides[i] *= new_stride;
        }
        return strides;
    }

    static CoreRangeSet calculate_core_range_set(
        TensorMemoryLayout memory_layout,
        ShardOrientation shard_orientation,
        Dim2d grid_offset,
        Dim2d grid_shape,
        Dim2d grid_extents) {
        TT_ASSERT(memory_layout != TensorMemoryLayout::HEIGHT_SHARDED || grid_shape.x == 1);
        TT_ASSERT(memory_layout != TensorMemoryLayout::WIDTH_SHARDED || grid_shape.y == 1);
        TT_ASSERT(memory_layout != TensorMemoryLayout::SINGLE_BANK || grid_shape.volume() == 1);
        TT_ASSERT(grid_shape.volume() <= grid_extents.volume());
        if (shard_orientation == ShardOrientation::COL_MAJOR) {
            std::swap(grid_shape.y, grid_shape.x);
        }
        TT_ASSERT(memory_layout != TensorMemoryLayout::BLOCK_SHARDED || grid_shape.y <= grid_extents.y);
        TT_ASSERT(memory_layout != TensorMemoryLayout::BLOCK_SHARDED || grid_shape.x <= grid_extents.x);

        std::set<CoreRange> core_ranges;
        auto dy = grid_shape.y / grid_extents.y;
        auto my = grid_shape.y % grid_extents.y;
        auto dx = grid_shape.x / grid_extents.x;
        auto mx = grid_shape.x % grid_extents.x;
        TT_ASSERT(dy == 0 or dx == 0);
        CoreCoord cursor(grid_offset.x, grid_offset.y);
        if (dy > 0) {
            CoreCoord start = cursor;
            CoreCoord end(dy, grid_extents.y);
            core_ranges.emplace(start, end);
            cursor = CoreCoord(dy + 1, grid_offset.y);
        } else if (dx > 0) {
            CoreCoord start = cursor;
            CoreCoord end(grid_extents.x, dx);
            core_ranges.emplace(start, end);
            cursor = CoreCoord(grid_offset.x, dx + 1);
        }

        CoreCoord start = cursor;
        CoreCoord end(mx, my);
        core_ranges.emplace(start, end);

        return CoreRangeSet(core_ranges);
    }

    inline void validate() const {
        TT_ASSERT(shape.size() >= 1);
        TT_ASSERT(strides.size() >= 2);
        TT_ASSERT(strides.size() == (shape.size() + 1));
        TT_ASSERT(strides.back() == 1);
        for (std::size_t i = 0; i < shape.size(); ++i) {
            TT_ASSERT(strides[i] > 0);
            TT_ASSERT(strides[i] % strides[i + 1] == 0);
            TT_ASSERT(strides[i] >= (shape[i] * strides[i + 1]));
        }
        Dim2d collapsed_shape = get_collapsed_shape2d();
        TT_ASSERT(collapsed_shape.y % shard_shape.dims.y == 0);
        TT_ASSERT(collapsed_shape.x % shard_shape.dims.x == 0);
    }

    inline Dim2d get_collapsed_shape2d() const {
        TT_ASSERT(strides.size() >= 2);
        auto y = strides[0] / strides[strides.size() - 2];
        auto x = strides[strides.size() - 2];
        return Dim2d(y, x);
    }

    inline Dim2d get_physical_shard_shape() const { return shard_shape.dims.align(tile_shape.dims); }

    inline GridShape get_grid_shape() const {
        Dim2d collapsed_shape = get_collapsed_shape2d();
        return GridShape(collapsed_shape.y / shard_shape.dims.y, collapsed_shape.x / shard_shape.dims.x);
    }

    inline std::uint32_t get_stride(int dim) const {
        if (dim < 0) {
            dim += strides.size();
        }
        TT_ASSERT(dim < strides.size());
        return strides[dim];
    }

    inline ::tt::tt_metal::MemoryConfig get_memory_config(
        ::tt::tt_metal::BufferType buffer_type,
        ::tt::tt_metal::TensorMemoryLayout memory_layout,
        ::tt::tt_metal::ShardOrientation shard_orientation,
        Dim2d grid_extents,
        Dim2d grid_offset = Dim2d(0, 0)) const {
        Dim2d physical_shard_shape = get_physical_shard_shape();
        GridShape grid_shape = get_grid_shape();
        auto core_range_set =
            calculate_core_range_set(memory_layout, shard_orientation, grid_offset, grid_shape.dims, grid_extents);
        ::tt::tt_metal::MemoryConfig memory_config;
        memory_config.memory_layout = memory_layout;
        memory_config.buffer_type = buffer_type;
        memory_config.shard_spec = ::tt::tt_metal::ShardSpec(
            core_range_set, {physical_shard_shape.y, physical_shard_shape.x}, shard_orientation, false);
        return memory_config;
    }
};

inline std::ostream &operator<<(std::ostream &os, Layout2::OOBVal v) {
    switch (v) {
        case Layout2::OOBVal::Any: os << "Any"; break;
        case Layout2::OOBVal::Zero: os << "Zero"; break;
        case Layout2::OOBVal::Infinity: os << "Infinity"; break;
        case Layout2::OOBVal::NegativeInfinity: os << "NegativeInfinity"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

inline std::ostream &operator<<(std::ostream &os, Dim2d dim2d) {
    os << "Dim2d(" << dim2d.y << ", " << dim2d.x << ")";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, Layout2 const &layout) {
    os << "Layout(";
    os << "shape=" << layout.shape;
    os << ", strides={";
    for (std::size_t i = 0; i < layout.strides.size(); ++i) {
        os << layout.strides[i];
        if (i + 1 < layout.strides.size()) {
            os << ", ";
        }
    }
    os << "}";
    os << ", shard_shape=" << layout.shard_shape.dims;
    os << ", tile_shape=" << layout.tile_shape.dims;
    os << ", oob_val=" << layout.oob_val;
    os << ")";
    return os;
}
}  // namespace ttnn
