#pragma once

#include "specfem/element_connections/tags.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/point/acceleration.hpp"
#include "specfem/point/displacement.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"

/**
 * @brief Element coupling configuration for multi-physics interfaces.
 *
 * Provides compile-time interface configuration for coupling different
 * physics media (elastic-acoustic, acoustic-elastic). Defines coupling
 * directions, flux schemes, and field type resolution through template
 * specializations.
 *
 */
namespace specfem::element_coupling {} // namespace specfem::element_coupling

#include "element_coupling/attributes.hpp"
#include "element_coupling/to_string.hpp"
