#include "utilities/strings.hpp"
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

} // namespace utilities
} // namespace specfem
