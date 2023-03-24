#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {
    // Allows support for tilizing A with DTX address map, untilize B, untilize output
    Tensor conv_as_large_bmm_single_block_single_core(const Tensor& a, const Tensor& b, bool untilize_out);
}

}
