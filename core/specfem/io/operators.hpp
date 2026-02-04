#pragma once
#include <string>

namespace specfem {
namespace io {

/**
 * @brief Tag type for write operations
 *
 * Used as a template parameter to select write-specific implementations.
 */
class write {};

/**
 * @brief Tag type for read operations
 *
 * Used as a template parameter to select read-specific implementations.
 */
class read {};

} // namespace io
} // namespace specfem
