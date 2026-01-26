#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace properties {

/**
 * @defgroup specfem_medium_properties_dim2_elastic_isotropic 2D Elastic
 * Isotropic Properties
 *
 */

/**
 * @ingroup specfem_medium_properties_dim2_elastic_isotropic
 * @brief Elastic isotropic material properties container.
 *
 * Stores material properties for elastic wave propagation in isotropic solids.
 * Supports both elastic and viscoelastic media types through template
 * specialization with SFINAE (enable_if constraint).
 *
 * **Material parameters:**
 * - `kappa`: Bulk modulus (resistance to compression)
 * - `mu`: Shear modulus (resistance to shear deformation)
 * - `rho`: Density (mass per unit volume)
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

  DATA_CONTAINER(kappa, mu, rho)
};

} // namespace properties

} // namespace medium
} // namespace specfem
