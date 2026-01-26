#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace kernels {

/**
 * @defgroup specfem_medium_kernels_dim2_elastic_anisotropic 2D Elastic
 * Anisotropic Misfit Kernels
 *
 */

/**
 * @ingroup specfem_medium_kernels_dim2_elastic_anisotropic
 * @brief Elastic anisotropic misfit kernels container (2D).
 *
 * Stores sensitivity kernels for seismic inversion of anisotropic elastic
 * parameters. Kernels quantify how changes in elastic stiffness tensor
 * components affect the seismic misfit, enabling gradient-based optimization.
 *
 * **Kernel types:**
 * - `rho`: Density kernel
 * - `c11`, `c13`, `c15`: Stiffness tensor kernels (row 1)
 * - `c33`, `c35`: Stiffness tensor kernels (row 3)
 * - `c55`: Shear stiffness kernel
 *
 * @tparam MediumTag Physical medium type (elastic, elastic_sh, elastic_psv)
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::element::medium_tag MediumTag>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  DATA_CONTAINER(rho, c11, c13, c15, c33, c35, c55)
};

} // namespace kernels

} // namespace medium
} // namespace specfem
