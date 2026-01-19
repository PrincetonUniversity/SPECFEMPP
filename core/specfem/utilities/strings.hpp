#pragma once

#include "specfem/macros.hpp"
#include <string>

namespace specfem {
namespace utilities {

// Convert integer to string with zero leading
std::string to_zero_lead(const int value, const int n_zero);

// Convert snake_case string to PascalCase
std::string snake_to_pascal(const std::string &str);

// convert string to lower case
std::string to_lower(const std::string &str);

BOOST_PP_SEQ_FOR_EACH(_DECLARE_CONFIG_STRING_FUNCTIONS, _, CONFIG_STRINGS)

} // namespace utilities
} // namespace specfem
