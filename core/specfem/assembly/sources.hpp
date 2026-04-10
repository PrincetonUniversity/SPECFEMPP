#pragma once

#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch/element_combinations.hpp"

namespace specfem::assembly {

/**
 * @brief Define the tag sets for source combinations in each dimension
 *
 * Specifies which medium, property, attenuation, boundary, and wavefield
 * combinations are valid for sources in 2D and 3D simulations.
 */
namespace sources_impl {
template <specfem::element::dimension_tag DimensionTag> struct SourceSets;

template <> struct SourceSets<specfem::element::dimension_tag::dim2> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  constexpr static auto medium_set =
      MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic, elastic_psv_t);
  constexpr static auto property_set =
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat);
  constexpr static auto attenuation_set =
      ATTENUATION_SET(none, constant_isotropic);
  constexpr static auto boundary_set = BOUNDARY_SET(
      none, acoustic_free_surface, stacey, composite_stacey_dirichlet);
  constexpr static auto wavefield_set =
      WAVEFIELD_SET(forward, backward, adjoint);
};

template <> struct SourceSets<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto medium_set = MEDIUM_SET(elastic, acoustic);
  constexpr static auto property_set = PROPERTY_SET(isotropic);
  constexpr static auto attenuation_set =
      ATTENUATION_SET(none, constant_isotropic);
  constexpr static auto boundary_set = BOUNDARY_SET(none);
  constexpr static auto wavefield_set =
      WAVEFIELD_SET(forward, backward, adjoint);
};
} // namespace sources_impl

/**
 * @brief Assembly-level source management for spectral element simulations
 *
 * This template class manages sources within assembled finite element meshes,
 * providing efficient access to source data for both host and device
 * computations. The sources are organized by medium type (elastic, acoustic,
 * poroelastic) and support time-dependent source time functions.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
struct sources; ///< Forward declaration of sources class

} // namespace specfem::assembly

// Include template specializations
#include "sources/dim2/sources.hpp"
#include "sources/dim3/sources.hpp"
