#pragma once

#include "specfem/macros.hpp"
#include <string>

namespace specfem {
namespace utilities {

/**
 * @brief Convert integer to string with leading zeros.
 *
 * Converts an integer to a string with zero-padding to ensure a minimum
 * number of digits. Useful for creating numbered filenames or identifiers.
 *
 * @param value Integer value to convert
 * @param n_zero Minimum number of digits in the output string
 * @return std::string Zero-padded string representation
 *
 * @code
 * to_zero_lead(42, 5);  // Returns "00042"
 * to_zero_lead(1234, 3); // Returns "1234" (no padding needed)
 * @endcode
 */
std::string to_zero_lead(const int value, const int n_zero);

/**
 * @brief Convert snake_case string to PascalCase.
 *
 * Transforms a string from snake_case (words separated by underscores) to
 * PascalCase (capitalized words without separators).
 *
 * @param str Input string in snake_case format
 * @return std::string String converted to PascalCase
 *
 * @code
 * snake_to_pascal("my_variable_name"); // Returns "MyVariableName"
 * @endcode
 */
std::string snake_to_pascal(const std::string &str);

/**
 * @brief Convert string to lowercase.
 *
 * @param str Input string
 * @return std::string Lowercase version of the input string
 *
 * @code
 * to_lower("HELLO World"); // Returns "hello world"
 * @endcode
 */
std::string to_lower(const std::string &str);

/**
 * @brief Strip leading and trailing whitespace from a string.
 *
 * @param str Input string
 * @return std::string String with leading/trailing whitespace removed
 *
 * @code
 * trim("  hello world  "); // Returns "hello world"
 * @endcode
 */
std::string trim(const std::string &str);

BOOST_PP_SEQ_FOR_EACH(_DECLARE_CONFIG_STRING_FUNCTIONS, _, CONFIG_STRINGS)

} // namespace utilities
} // namespace specfem
