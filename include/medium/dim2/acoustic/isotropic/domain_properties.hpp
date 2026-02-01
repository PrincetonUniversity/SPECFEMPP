#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {
namespace properties {

/**
 * @brief Acoustic isotropic material properties container.
 *
 * Stores material properties for acoustic wave propagation in fluids.
 * Acoustic media support only compressional waves (P-waves) with no
 * shear wave propagation, suitable for modeling fluids and gases.
 *
 * **Material parameters:**
 * - `rho_inverse`: Inverse density (1/ρ) for computational efficiency
 * - `kappa`: Bulk modulus (resistance to compression)
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::dimension::type DimensionTag>
struct data_container<DimensionTag, specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic> {

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rho_inverse, kappa)
};
} // namespace properties

} // namespace medium
} // namespace specfem
