#pragma once

#include "specfem/medium_container/impl/domain_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem::medium_container::properties {

/**
 * @defgroup specfem_medium_properties_dim3_elastic_anisotropic 3D Elastic
 * Anisotropic Properties
 *
 */

/**
 * @ingroup specfem_medium_properties_dim3_elastic_anisotropic
 * @brief Elastic anisotropic material properties container (3D).
 *
 * Stores the 21 independent entries of the symmetric stiffness matrix in
 * Voigt notation and density. Storage is allocated only for elements in the
 * anisotropic elastic property group.
 *
 * @tparam MediumTag Physical medium type; must be elastic.
 * @see DATA_CONTAINER macro for generated storage and access functions.
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

  DATA_CONTAINER(c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33,
                 c34, c35, c36, c44, c45, c46, c55, c56, c66, rho)
};

} // namespace specfem::medium_container::properties
