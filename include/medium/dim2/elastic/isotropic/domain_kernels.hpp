#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {

/**
 * @defgroup specfem_medium_kernels_dim2_elastic_isotropic 2D Elastic Isotropic
 * Misfit Kernels
 *
 */

/**
 * @ingroup specfem_medium_kernels_dim2_elastic_isotropic
 * @brief Elastic isotropic misfit kernels container.
 *
 * Stores sensitivity kernels for seismic inversion of elastic material
 * parameters. Kernels quantify how changes in material properties affect the
 * seismic misfit, enabling gradient-based optimization.
 *
 * **Kernel types:**
 * - `rho`: Density kernel
 * - `mu`: Shear modulus kernel
 * - `kappa`: Bulk modulus kernel
 * - `rhop`: Density perturbation kernel
 * - `alpha`: P-wave velocity kernel
 * - `beta`: S-wave velocity kernel
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam MediumTag Physical medium type (elastic, elastic_sh, elastic_psv)
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct data_container<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;

  DATA_CONTAINER(rho, mu, kappa, rhop, alpha, beta)
};

} // namespace kernels
} // namespace medium
} // namespace specfem
