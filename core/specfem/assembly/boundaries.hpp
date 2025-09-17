#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::boundaries_impl {

template <specfem::dimension::type DimensionTag> struct acoustic_free_surface;

template <specfem::dimension::type DimensionTag> struct stacey;

} // namespace specfem::assembly::boundaries_impl

namespace specfem::assembly {

/**
 * @brief Class representing the boundaries in the assembly.
 *
 * This class provides an interface for working with boundaries in the assembly.
 */
template <specfem::dimension::type DimensionTag> class boundaries;

} // namespace specfem::assembly

#include "boundaries/dim2/boundaries.hpp"
#include "boundaries/dim2/impl/acoustic_free_surface.hpp"
#include "boundaries/dim2/impl/stacey.hpp"
