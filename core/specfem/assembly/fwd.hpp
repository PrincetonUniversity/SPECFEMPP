#pragma once

#include "specfem/element/tags.hpp"

namespace specfem::assembly {

/**
 * @brief Forward declaration of the assembly data container
 *
 * Use this header when only a reference or pointer to assembly is needed in
 * a function declaration. Include "specfem/assembly.hpp" in implementation
 * files where assembly members are accessed.
 */
template <specfem::element::dimension_tag DimensionTag> struct assembly;

} // namespace specfem::assembly
