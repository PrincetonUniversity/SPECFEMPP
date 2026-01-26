#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {

/**
 * @brief Poroelastic isotropic misfit kernels container.
 *
 * Stores sensitivity kernels for seismic inversion of porous media parameters.
 * Kernels quantify how changes in material properties affect the seismic
 * wavefield, enabling gradient-based optimization.
 *
 * **Kernel types:**
 * - `rhot`, `rhof`: Density kernels (total, fluid)
 * - `eta`: Fluid viscosity kernel
 * - `sm`, `mu_fr`: Shear modulus kernels
 * - `B`, `C`, `M`: Biot elastic constant kernels
 * - `phi`: Porosity kernel
 * - `cpI`, `cpII`, `cs`: Wave velocity kernels (P1, P2, S)
 * - `ratio`: Permeability ratio kernel
 * - `rhob`, `rhofb`, `mu_frb`, `rhobb`, `rhofbb`, `phib` : Bulk parameters
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

  DATA_CONTAINER(rhot, rhof, eta, sm, mu_fr, B, C, M, mu_frb, rhob, rhofb, phi,
                 cpI, cpII, cs, rhobb, rhofbb, ratio, phib);
};
} // namespace kernels

} // namespace medium
} // namespace specfem
