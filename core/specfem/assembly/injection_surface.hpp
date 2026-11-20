#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Quadrature points of the wavefield-injection surface declared by the
 * mesher
 *
 * General template declaration. The specializations hold, for every GLL point
 * on an injection face, the geometry an injection method needs, independent
 * of the plugin and of the quantities it provides.
 *
 * @see
 * specfem::assembly::InjectionSurface<specfem::element::dimension_tag::dim3>
 */
template <specfem::element::dimension_tag DimensionTag> struct InjectionSurface;

} // namespace specfem::assembly

#include "injection_surface/dim3/injection_surface.hpp"
