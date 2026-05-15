#pragma once

#include <string>

namespace specfem {
namespace enums {

/**
 * @brief Supported source file formats.
 */
enum class source_format { YAML, CMTSOLUTION, FORCESOLUTION };

/**
 * @brief A single source file entry with its format and path.
 */
struct source_file_entry {
  source_format format;
  std::string file_path;
};

/**
 * @brief Convert source_format to its string representation.
 */
std::string to_string(const source_format &fmt);

} // namespace enums
} // namespace specfem
