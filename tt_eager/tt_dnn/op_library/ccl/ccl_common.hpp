// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <cstdint>


namespace tt {
namespace tt_metal {
namespace ccl {

struct CclTensorSlicer {
    CclTensorSlicer(
        Shape tensor_shape,
        Shape dim_slice_factors,
        Shape page_shape
    ) :
        tensor_shape(tensor_shape),
        dim_slice_factors_per_rank(dim_slice_factors),
        page_shape(page_shape)
    {
        TT_ASSERT(tensor_shape.rank() == dim_slice_factors.rank(),
                  "Tensor shape and dim slice factors must have the same size");
        TT_ASSERT(std::all_of(dim_slice_factors.begin(), dim_slice_factors.end(), [](uint32_t factor) { return factor > 0; }),
                  "All factors must be greater than 0");
        TT_ASSERT(page_shape.rank() == 2 || page_shape.rank() == tensor_shape.rank(),
                  "Page shape must have rank 2 or the same rank as the tensor shape");

    }

    Shape const tensor_shape;
    Shape const dim_slice_factors_per_rank;
    Shape const page_shape;

    Shape rank_slice_shape;
};

class InterleavedRingReduceScatterTensorSlicer : public CclTensorSlicer {
   public:
    InterleavedRingReduceScatterTensorSlicer() :
        CclTensorSlicer() {

        }


};



// To be replaced by the CclTensorSlicer class, which should be reusable between sharded and interleaved
// specs and also provides a simpler interface to reason about
struct LegacyCclTensorSlicer {
    LegacyCclTensorSlicer() :
        input_page_size(0),
        num_rows(0),
        num_cols(0),
        row_offset(0),
        col_offset(0),
        num_tiles(0),
        input_start_page_idx(0),
        output_addr_offset(0),
        col_idx(0),
        row_idx(0),
        output_page_offset(0),
        output_start_page_idx(0),
        output_start_addr_offset(0),
        row_major(false),
        slice_dim_is_width(false),
        is_sharded(false) {}

    LegacyCclTensorSlicer(
        uint32_t input_page_size,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t row_offset,
        uint32_t col_offset,
        uint32_t num_tiles,
        uint32_t input_start_page_idx,
        uint32_t output_addr_offset,
        uint32_t col_idx,
        uint32_t row_idx,
        uint32_t output_page_offset,
        uint32_t output_start_page_idx,
        uint32_t output_start_addr_offset,
        bool row_major,
        bool slice_dim_is_width,
        bool is_sharded) :
        input_page_size(input_page_size),
        num_rows(num_rows),
        num_cols(num_cols),
        row_offset(row_offset),
        col_offset(col_offset),
        num_tiles(num_tiles),
        input_start_page_idx(input_start_page_idx),
        output_addr_offset(output_addr_offset),
        col_idx(col_idx),
        row_idx(row_idx),
        output_page_offset(output_page_offset),
        output_start_page_idx(output_start_page_idx),
        output_start_addr_offset(output_start_addr_offset),
        row_major(row_major),
        slice_dim_is_width(slice_dim_is_width),
        is_sharded(is_sharded) {}

    virtual void increment(uint32_t num_pages) = 0;

    uint32_t input_page_size;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t row_offset;
    uint32_t col_offset;
    uint32_t num_tiles;
    uint32_t input_start_page_idx;
    uint32_t output_addr_offset;
    uint32_t col_idx;
    uint32_t row_idx;
    uint32_t output_page_offset;
    uint32_t output_start_page_idx;
    uint32_t output_start_addr_offset;
    bool row_major;
    bool slice_dim_is_width;
    bool is_sharded;
};

class InterleavedRingAllGatherTensorSlicer : public LegacyCclTensorSlicer {
   public:
    InterleavedRingAllGatherTensorSlicer (
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        int slice_dim,
        uint32_t slice_idx
    ) : LegacyCclTensorSlicer() {

        this->row_major = input_tensor.get_layout() == Layout::ROW_MAJOR;
        this->slice_dim_is_width = input_tensor.get_legacy_shape().rank() - 1 == slice_dim;
        this->is_sharded = input_tensor.is_sharded();

        int32_t shard_size_in_bytes = is_sharded ?
            (input_tensor.buffer()->page_size() * input_tensor.buffer()->shard_spec().tensor2d_shape[0] * input_tensor.buffer()->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores() :
            -1;
        this->input_page_size = is_sharded ? shard_size_in_bytes : input_tensor.buffer()->page_size();;
        if (row_major) {
            num_cols = input_tensor.get_legacy_shape()[-1];
            auto input_shape = input_tensor.get_legacy_shape();
            auto output_shape = output_tensor.get_legacy_shape();
            num_rows = std::accumulate(input_shape.begin() + slice_dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>());
            row_offset = std::accumulate(output_shape.begin() + slice_dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) - num_rows;
        } else {
            num_cols = input_tensor.get_legacy_shape()[-1] / tt::constants::TILE_WIDTH;
            auto input_shape = input_tensor.get_legacy_shape();
            auto output_shape = output_tensor.get_legacy_shape();
            uint32_t num_output_cols = output_tensor.get_legacy_shape()[-1] / tt::constants::TILE_WIDTH;
            num_rows = std::accumulate(input_shape.begin() + slice_dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>()) / tt::constants::TILE_HEIGHT;
            row_offset = (std::accumulate(output_shape.begin() + slice_dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) / tt::constants::TILE_HEIGHT - num_rows) * num_output_cols;
            col_offset = num_output_cols - num_cols;
            num_tiles = num_rows * num_cols;
        }

        if (row_major) {
            if (slice_dim_is_width) {
                output_addr_offset = input_page_size;
            } else {
                output_page_offset = num_rows;
            }
        } else {
            if (slice_dim_is_width) {
                output_page_offset = num_cols;
            } else {
                output_page_offset = num_tiles;
            }
        }
        output_start_page_idx = slice_idx/*ring_index*/ * output_page_offset;
        output_start_addr_offset = slice_idx/*ring_index*/ * output_addr_offset;
    }

    virtual void increment(uint32_t num_pages) override {
        // uint32_t pages_per_worker = num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b);
        if (is_sharded) {
            // nothing to do here - is handled by
        } else {
            // Only for interleaved
            if (num_pages/*pages_per_worker*/ > 0) {
                if (row_major) {
                    uint32_t num_rows_shifted = row_idx + num_pages/*pages_per_worker*/;
                    uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += num_pages/*pages_per_worker*/ + num_blocks_shifted * row_offset;
                    row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
                } else {
                    uint32_t num_cols_shifted = col_idx + num_pages/*pages_per_worker*/;
                    uint32_t num_rows_shifted = num_cols_shifted / num_cols;
                    uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += num_pages/*pages_per_worker*/ + num_rows_shifted * col_offset + num_blocks_shifted * row_offset;
                    col_idx = num_cols_shifted % num_cols;
                    row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
                }
            }
            input_start_page_idx += num_pages/*pages_per_worker*/;
        }
    }
};



KernelHandle generate_edm_kernel(
    tt_metal::Program &program,
    Device const* device,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    NOC noc_id);

void generate_edm_kernels_for_ring_or_linear_topology(
    tt_metal::Program &program,
    Device const* device,
    std::vector<ccl::EriscDatamoverBuilder> const& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id,
    // TODO: move to linear/ring topology specific config
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    bool is_linear);

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode);

} // namespace ccl
} // namespace tt_metal
} // namespace tt
