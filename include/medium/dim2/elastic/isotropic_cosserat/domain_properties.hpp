#pragma once

#include "medium/impl/data_container.hpp"
#include "specfem/point.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem::medium::properties {

/**
 * @defgroup specfem_medium_properties_dim2_elastic_isotropic_cosserat 2D
 * Elastic Isotropic Cosserat Properties
 *
 */

/**
 * @ingroup specfem_medium_properties_dim2_elastic_isotropic_cosserat Specfem
 * Medium Properties for 2D Elastic Isotropic Cosserat Media
 *
 * @brief Data container to hold properties of 2D elastic isotropic Cosserat
 * media at a quadrature point
 *
 * @tparam MediumTag The type of the medium
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 * @tparam Enable SFINAE type to enable this specialization only for elastic
 * media
 *
 * Parameters:
 * - `rho`: Density @f$ \rho @f$
 * - `kappa`: Bulk modulus @f$ \kappa @f$
 * - `mu`: Shear modulus @f$ \mu @f$
 * - `nu`: Symmetry breaking coupling modulus @f$ \nu @f$
 * - `j`: Inertia density @f$ j @f$
 * - `lambda_c`: Coupling bulk modulus @f$ \lambda_c @f$
 * - `mu_c`: Coupling shear modulus @f$ \mu_c @f$
 * - `nu_c`: Coupling symmetry breaking modulus @f$ \nu_c @f$
 *
 * @see DATA_CONTAINER macro for details on generated members and methods.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct data_container<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic_cosserat,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {

  constexpr static auto dimension_tag =
      DimensionTag;                             ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic_cosserat; ///< Property tag

  DATA_CONTAINER(rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c)
};

} // namespace specfem::medium::properties
