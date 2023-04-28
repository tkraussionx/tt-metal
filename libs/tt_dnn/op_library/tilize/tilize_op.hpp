#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor tilize (const Tensor &a);
Tensor tilize_with_zero_padding (const Tensor &a);
Tensor tilize_conv_activation (const Tensor &a, vector<int> conv_params, int conv_output_channels);
}  // namespace tt_metal

}  // namespace tt
