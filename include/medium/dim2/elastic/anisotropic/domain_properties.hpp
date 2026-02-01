#pragma once

#include "medium/impl/data_container.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

namespace properties {

/**
 * @defgroup specfem_medium_properties_dim2_elastic_anisotropic 2D Elastic
 * Anisotropic Properties
 *
 */

/**
 * @ingroup specfem_medium_properties_dim2_elastic_anisotropic
 * @brief Elastic anisotropic material properties container (2D).
 *
 * Stores material properties for elastic wave propagation in anisotropic
 * solids. Uses full elastic stiffness tensor representation for 2D anisotropic
 * media with SFINAE constraint for elastic medium types.
 *
 * **Material parameters:**
 * - `c11`, `c13`, `c15`: Elastic stiffness tensor components (row 1)
 * - `c33`, `c35`: Elastic stiffness tensor components (row 3)
 * - `c55`: Elastic stiffness tensor component (shear)
 * - `c12`, `c23`, `c25`: Additional stiffness tensor components
 * - `rho`: Density (mass per unit volume)
 *
 * @tparam MediumTag Physical medium type (elastic, elastic_sh, elastic_psv)
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

  DATA_CONTAINER(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho)
};

} // namespace properties

} // namespace medium
} // namespace specfem
