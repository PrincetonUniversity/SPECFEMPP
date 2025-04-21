#pragma once

#include <string>

namespace specfem {
namespace utilities {

// Convert integer to string with zero leading
std::string to_zero_lead(const int value, const int n_zero);

// Convert snake_case string to PascalCase
std::string snake_to_pascal(const std::string &str);

// convert string to lower case
std::string to_lower(const std::string &str);

// Check if string is hdf5
bool is_hdf5_string(const std::string &str);

// Check if string is ascii
bool is_ascii_string(const std::string &str);

// check if string indicates P-SV wave
bool is_psv_string(const std::string &str);

// check if string indicates SH wave
bool is_sh_string(const std::string &str);

// check if string indicates TE wave
bool is_te_string(const std::string &str);

// check if string is jpg
bool is_jpg_string(const std::string &str);

// check if string is png
bool is_png_string(const std::string &str);

// check if string is sac
bool is_sac_string(const std::string &str);

// check if string is seismic unix
bool is_su_string(const std::string &str);

// check if string indicates forward simulation
bool is_forward_string(const std::string &str);

// check if string indicates combined simulation
bool is_combined_string(const std::string &str);

// check if string indicates backward simulation
bool is_backward_string(const std::string &str);

// check if string indicates adjoint simulation
bool is_adjoint_string(const std::string &str);

// check if string is GLL-4
bool is_gll4_string(const std::string &str);

// check if string is GLL-7
bool is_gll7_string(const std::string &str);

// check if string indicates displacement
bool is_displacement_string(const std::string &str);

// check if string indicates velocity
bool is_velocity_string(const std::string &str);

// check if string indicates acceleration
bool is_acceleration_string(const std::string &str);

// check if string indicates pressure
bool is_pressure_string(const std::string &str);

// check if string is Newmark
bool is_newmark_string(const std::string &str);

// check if string is on_screen
bool is_onscreen_string(const std::string &str);

} // namespace utilities
} // namespace specfem
