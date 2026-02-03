#include "strings.hpp"
#include <algorithm>

namespace specfem {
namespace utilities {

// Convert integer to string with zero leading
std::string to_zero_lead(const int value, const int n_zero) {
  auto old_str = std::to_string(value);
  int n_zero_fix =
      n_zero - std::min(n_zero, static_cast<int>(old_str.length()));
  auto new_str = std::string(n_zero_fix, '0') + old_str;
  return new_str;
}

// Convert snake_case string to PascalCase
std::string snake_to_pascal(const std::string &str) {
  std::string result;
  bool capitalizeNext = true; // Capitalize the first character

  for (char ch : str) {
    if (ch == '_') {
      capitalizeNext = true;
    } else if (capitalizeNext) {
      result += std::toupper(ch);
      capitalizeNext = false;
    } else {
      result += ch;
    }
  }
  return result;
}

// convert string to lower case
std::string to_lower(const std::string &str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower_str;
}

BOOST_PP_SEQ_FOR_EACH(_DEFINE_CONFIG_STRING_FUNCTIONS, _, CONFIG_STRINGS)

} // namespace utilities
} // namespace specfem
