#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace properties {

/**
 * @brief Poroelastic isotropic material properties container.
 *
 * Specialization for fluid-saturated porous media using Biot's theory.
 * Stores material properties for modeling wave propagation in porous media
 * with fluid-solid coupling effects.
 *
 * **Physical parameters:**
 * - `phi`: Porosity (void fraction)
 * - `rho_s`, `rho_f`: Solid and fluid densities
 * - `tortuosity`: Pore geometry factor
 * - `mu_G`: Shear modulus of solid matrix
 * - `H_Biot`, `C_Biot`, `M_Biot`: Biot elastic constants
 * - `permxx`, `permxz`, `permzz`: Permeability tensor components
 * - `eta_f`: Fluid viscosity
 *
 * Generated storage provides device/host DomainViews for all parameters
 * organized by quadrature points within spectral elements.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::dimension::type DimensionTag>
struct data_container<DimensionTag, specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic> {
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = specfem::element::medium_tag::poroelastic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot, M_Biot,
                 permxx, permxz, permzz, eta_f)
};

} // namespace properties
} // namespace medium
} // namespace specfem
