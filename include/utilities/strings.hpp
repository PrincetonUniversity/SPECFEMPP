#pragma once

#include <string>

namespace specfem {
namespace utilities {

// Convert integer to string with zero leading
std::string to_zero_lead(const int value, const int n_zero);

// Convert snake_case string to PascalCase
std::string snake_to_pascal(const std::string &str);

} // namespace utilities
} // namespace specfem
