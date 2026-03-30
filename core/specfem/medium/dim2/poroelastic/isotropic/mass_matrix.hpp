#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_dim2_compute_mass_matrix_poroelastic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_mass_matrix_poroelastic
 * @brief Compute mass matrix inverse for 2D poroelastic isotropic media.
 *
 * Implements Biot's theory mass matrix for fluid-saturated porous media
 * with solid-fluid coupling. Computes effective densities accounting for
 * fluid-solid interactions and tortuosity effects.
 *
 * **Mass matrix equations:**
 *
 * Solid component:
 * \f$ M_s = \bar{\rho} - \frac{\phi \rho_f}{\alpha} \f$
 *
 * Fluid component:
 * \f$ M_f = \frac{\rho_f \alpha \bar{\rho} - \phi \rho_f^2}{\phi \bar{\rho}}
 * \f$
 *
 * **Physical parameters:**
 * - \f$ \bar{\rho} \f$: Bulk density of saturated medium
 * - \f$ \rho_f \f$: Fluid density
 * - \f$ \phi \f$: Porosity
 * - \f$ \alpha \f$: Tortuosity (pore geometry factor)
 *
 * Returns 4 components for 2D poroelastic system: [M_s, M_s, M_f, M_f]
 * where M_s and M_f are solid and fluid mass matrix components.
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param properties Poroelastic material properties
 * @return Inverse mass matrix components for explicit time integration
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim2 &&
            Tags::medium_tag == specfem::element::medium_tag::poroelastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_FUNCTION specfem::point::mass_inverse<Tags>
impl_mass_matrix_component(const specfem::point::properties<Tags> &properties) {

  const auto solid_component =
      (properties.rho_bar() -
       properties.phi() * properties.rho_f() / properties.tortuosity());
  const auto fluid_component =
      (properties.rho_f() * properties.tortuosity() * properties.rho_bar() -
       properties.phi() * properties.rho_f() * properties.rho_f()) /
      (properties.phi() * properties.rho_bar());

  return { solid_component, solid_component, fluid_component, fluid_component };
}

} // namespace medium_physics
} // namespace specfem
