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

// Check if string is hdf5
bool is_hdf5_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "hdf5" || str_lower == "h5";
}

// Check if string is ascii
bool is_ascii_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "ascii" || str_lower == "txt";
}

// check if string indicates P-SV wave
bool is_psv_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "psv" || str_lower == "p_sv" || str_lower == "p-sv";
}

// check if string indicates SH wave
bool is_sh_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "sh";
}

// check if string indicates TE wave
bool is_te_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "te";
}

// check if string is jpg
bool is_jpg_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "jpg" || str_lower == "jpeg";
}

// check if string is png
bool is_png_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "png";
}

// check if string is sac
bool is_sac_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "sac";
}

// check if string is seismic unix
bool is_su_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "su" || str_lower == "seismic_unix" ||
         str_lower == "seismic-unix";
}

// check if string indicates forward simulation
bool is_forward_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "forward";
}

// check if string indicates combined simulation
bool is_combined_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "combined";
}

// check if string indicates backward simulation
bool is_backward_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "backward";
}

// check if string indicates adjoint simulation
bool is_adjoint_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "adjoint";
}

// check if string is GLL-4
bool is_gll4_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "gll4" || str_lower == "gll-4" || str_lower == "gll_4";
}

// check if string is GLL-7
bool is_gll7_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "gll7" || str_lower == "gll-7" || str_lower == "gll_7";
}

// check if string indicates displacement
bool is_displacement_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "displacement" || str_lower == "disp" ||
         str_lower == "displ" || str_lower == "d";
}

// check if string indicates velocity
bool is_velocity_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "velocity" || str_lower == "vel" || str_lower == "v" ||
         str_lower == "veloc";
}

// check if string indicates acceleration
bool is_acceleration_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "acceleration" || str_lower == "accel" ||
         str_lower == "acc" || str_lower == "a";
}

// check if string indicates pressure
bool is_pressure_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "pressure" || str_lower == "pres" || str_lower == "p";
}

// check if string is Newmark
bool is_newmark_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "newmark";
}

// check if string is on_screen
bool is_onscreen_string(const std::string &str) {
  const auto str_lower = to_lower(str);
  return str_lower == "onscreen" || str_lower == "on-screen" ||
         str_lower == "on_screen";
}

} // namespace utilities
} // namespace specfem
