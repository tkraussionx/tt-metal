#pragma once
#include "common/utils.hpp"

namespace tt {
namespace test_utils {
/// @brief generic vector printer
/// @tparam T
/// @param const std::vector<T>& vec
template <typename T>
void print_vector(const std::vector<T>& vec) {
    int idx = 0;
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec.at(i) << ", ";
    }
    std::cout << std::endl;
}
/// @brief generic vector printer in fixed row widths
/// @tparam T
/// @param const std::vector<T>& vec
template <typename T>
void print_vector_fixed_el_per_row(const std::vector<T>& vec, const unsigned int numel_per_row) {
    int idx = 0;
    for (int i = 0; i < vec.size(); i++) {
        if ((i%numel_per_row) == 0) {
            std::cout << std::endl;
        }
        std::cout << vec.at(i) << ", ";
    }
    std::cout << std::endl;
}
}  // namespace test_utils
}  // namespace tt
