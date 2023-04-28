#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor reader_writer_op_multi_blocks(const Tensor &a, vector<uint32_t> address_map, uint32_t num_blocks, uint32_t block_size);
}  // namespace tt_metal

}  // namespace tt
