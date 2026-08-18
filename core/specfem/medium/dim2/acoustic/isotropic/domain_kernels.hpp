#pragma once

#include "specfem/medium_container/impl/domain_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem::medium_container::kernels {

/**
 * @brief Acoustic isotropic misfit kernels container.
 *
 * Stores sensitivity kernels for seismic inversion of acoustic medium
 * parameters. Kernels quantify how changes in acoustic properties affect the
 * seismic misfit, enabling gradient-based optimization in acoustic full
 * waveform inversion.
 *
 * **Kernel types:**
 * - `rho`: Density kernel
 * - `kappa`: Bulk modulus kernel
 * - `rhop`: Density perturbation kernel
 * - `alpha`: P-wave velocity kernel
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::element::dimension_tag DimensionTag>
struct data_container<DimensionTag, specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic> {
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rho, kappa, rhop, alpha)
};

} // namespace specfem::medium_container::kernels
