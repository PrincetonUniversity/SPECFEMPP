#pragma once

#include "specfem/medium_container/impl/domain_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem::medium_container::kernels {

/**
 * @brief Misfit kernels for a 3D fully anisotropic elastic medium.
 *
 * Stores the absolute-parameter density kernel and the 21 independent entries
 * of the symmetric stiffness kernel in Voigt notation.
 *
 * @tparam MediumTag Physical medium type; must be elastic.
 */
template <specfem::element::medium_tag MediumTag>
struct data_container<
    specfem::element::dimension_tag::dim3, MediumTag,
    specfem::element::property_tag::anisotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value>> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;

  DATA_CONTAINER(rho, c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26,
                 c33, c34, c35, c36, c44, c45, c46, c55, c56, c66)
};

} // namespace specfem::medium_container::kernels
