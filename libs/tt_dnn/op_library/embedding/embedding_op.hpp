#pragma once

#include "libs/tensor/tensor.hpp"

namespace tt {
namespace tt_metal {

Tensor embedding_single_core(const Tensor &weights, const Tensor &in_data);
Tensor embedding(const Tensor &weights, const Tensor &in_data);

}
}
