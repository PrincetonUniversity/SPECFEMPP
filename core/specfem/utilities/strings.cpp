#include "strings.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

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

// Strip leading and trailing whitespace
std::string trim(const std::string &str) {
  auto start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

// Format a distance in metres as whole km, m, and mm for readability.
std::string format_distance(type_real metres) {
  const int km = static_cast<int>(metres / 1000);
  const int m = static_cast<int>(metres) % 1000;
  const int mm = static_cast<int>(metres * 1000) % 1000;
  return std::to_string(km) + " km, " + std::to_string(m) + " m, " +
         std::to_string(mm) + " mm";
}

// Format a floating-point value in scientific notation (pre-C++20 equivalent of
// std::format("{:.{}e}", value, precision)).
std::string format_scientific(type_real value, int precision) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(precision) << value;
  return oss.str();
}

BOOST_PP_SEQ_FOR_EACH(_DEFINE_CONFIG_STRING_FUNCTIONS, _, CONFIG_STRINGS)

} // namespace utilities
} // namespace specfem
