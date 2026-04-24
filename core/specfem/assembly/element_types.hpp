#pragma once

#include "element_types/impl.hpp"
#include "specfem/enums.hpp"

namespace specfem::assembly {

namespace element_types_impl {
template <specfem::element::dimension_tag DimensionTag> struct ElementSets;

template <> struct ElementSets<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr static auto medium_set =
      MEDIUM_SET(elastic_psv, elastic_sh, elastic_psv_t, acoustic, poroelastic,
                 electromagnetic_te);
  constexpr static auto property_set =
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
  constexpr static auto boundary_set = BOUNDARY_SET(
      none, stacey, acoustic_free_surface, composite_stacey_dirichlet);
  constexpr static auto attenuation_set =
      ATTENUATION_SET(none, constant_isotropic);
};

template <> struct ElementSets<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto medium_set =
      MEDIUM_SET(elastic, acoustic, elastic_spin);
  constexpr static auto property_set =
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
  constexpr static auto boundary_set = BOUNDARY_SET(none);
  constexpr static auto attenuation_set =
      ATTENUATION_SET(none, constant_isotropic);
};
} // namespace element_types_impl

/**
 * @brief Spectral element type classification and indexing container
 *
 * This class provides storage and management for element type information
 * in spectral element meshes, including medium types (elastic, acoustic,
 * poroelastic), material properties (isotropic, anisotropic, Cosserat),
 * and boundary conditions.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag>
using element_types = element_types_impl::element_types_base<
    element_types_impl::ElementSets<DimensionTag> >;
} // namespace specfem::assembly
