#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "point/sources.hpp"
#include "source/source.hpp"
#include "source_medium.hpp"

namespace specfem {
namespace compute {
struct sources {
private:
  using IndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using MediumTagViewType =
      Kokkos::View<specfem::element::medium_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store medium tags

  using WavefieldTagViewType =
      Kokkos::View<specfem::wavefield::simulation_field *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type
                                                   ///< to store wavefield
                                                   ///< tags

  using BoundaryTagViewType =
      Kokkos::View<specfem::element::boundary_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type
                                                   ///< to store boundary
                                                   ///< tags

  using PropertyTagViewType =
      Kokkos::View<specfem::element::property_tag *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type
                                                   ///< to store property
                                                   ///< tags

public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  sources() = default;

  /**
   * @brief Generate source information for a given mesh
   *
   * @param sources Vector of sources read from sources file
   * @param mesh Finite element mesh information
   * @param partial_derivatives Partial derivatives for every quadrature point
   * @param properties Material properties for every quadrature point
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  sources(
      const std::vector<std::shared_ptr<specfem::sources::source> > &sources,
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types, const type_real t0,
      const type_real dt, const int nsteps);
  ///@}

  /**
   * @brief Get the indices of spectral elements for all sources of a given
   * medium and wavefield type on the host
   *
   * @param medium Medium tag of the sources
   * @param wavefield Wavefield tag of the sources
   * @return Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> View of the
   * indices of sources of the given type
   */
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> get_elements_on_host(
      const specfem::element::medium_tag medium,
      const specfem::element::property_tag property,
      const specfem::element::boundary_tag boundary,
      const specfem::wavefield::simulation_field wavefield) const;

  /**
   * @brief Get the indices of spectral elements for all sources of a given
   * medium and wavefield type on the device
   *
   * @param medium Medium tag of the sources
   * @param wavefield Wavefield tag of the sources
   * @return Kokkos::View<int *, Kokkos::DefaultExecutionSpace> View of the
   * indices of sources of the given type
   */
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> get_elements_on_device(
      const specfem::element::medium_tag medium,
      const specfem::element::property_tag property,
      const specfem::element::boundary_tag boundary,
      const specfem::wavefield::simulation_field wavefield) const;

  /**
   * @brief Update the current time step
   *
   * The timestep is used when accessing the source time function using
   * `load_on_device` functions
   *
   * @param timestep Current time step
   */
  void update_timestep(const int timestep) { this->timestep = timestep; }

private:
  int nspec;                                 ///< Number of spectral elements
  IndexViewType source_domain_index_mapping; ///< Mapping for every spectral
                                             ///< element where source is
                                             ///< located to local index within
                                             ///< source_medium
  IndexViewType::HostMirror
      h_source_domain_index_mapping; ///< Host mirror of
                                     ///< source_domain_index_mapping
  MediumTagViewType medium_types;    ///< Medium type for every spectral element
  MediumTagViewType::HostMirror h_medium_types; ///< Host mirror of
                                                ///< medium_types
  WavefieldTagViewType wavefield_types; ///< Wavefield on which source is
                                        ///< applied
  WavefieldTagViewType::HostMirror h_wavefield_types; ///< Host mirror of
                                                      ///< wavefield_types
  BoundaryTagViewType boundary_types; ///< Boundary type for every spectral
                                      ///< element
  BoundaryTagViewType::HostMirror h_boundary_types; ///< Host mirror of
                                                    ///< boundary_types
  PropertyTagViewType property_types; ///< Property type for every spectral
                                      ///< element
  PropertyTagViewType::HostMirror h_property_types; ///< Host mirror of
                                                    ///< property_types

#define SOURCE_MEDIUM_DECLARATION(DIMENSION_TAG, MEDIUM_TAG)                   \
  specfem::compute::impl::source_medium<GET_TAG(DIMENSION_TAG),                \
                                        GET_TAG(MEDIUM_TAG)>                   \
      CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),                    \
                           GET_NAME(MEDIUM_TAG));

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(SOURCE_MEDIUM_DECLARATION,
                                 WHERE(DIMENSION_TAG_DIM2)
                                     WHERE(MEDIUM_TAG_ELASTIC,
                                           MEDIUM_TAG_ACOUSTIC))

#undef SOURCE_MEDIUM_DECLARATION

  int timestep; ///< Current time step

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void
  load_on_device(const IndexType index,
                 const specfem::compute::sources &sources,
                 PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend void load_on_host(const IndexType index,
                           const specfem::compute::sources &sources,
                           PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void
  store_on_device(const IndexType index, const PointSourceType &point_source,
                  const specfem::compute::sources &sources);

  template <typename IndexType, typename PointSourceType>
  friend void store_on_host(const IndexType index,
                            const PointSourceType &point_source,
                            const specfem::compute::sources &sources);
}; // namespace compute

/**
 * @brief Load source information on device at the given index
 *
 * Loads source information on device at the given index. Make sure you set
 the
 * correct timestep using `sources.update_timestep` before calling this
 * function.
 *
 * @tparam IndexType Point index type @ref specfem::point::index
 * @tparam PointSourceType Point source type @ref specfem::point::source
 * @param index Spectral element index to load source information
 * @param sources Source information for the domain
 * @param point_source Point source object to load source information into
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void
load_on_device(const IndexType index, const specfem::compute::sources &sources,
               PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      PointSourceType::is_point_source,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension == specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

  static_assert(
      ((PointSourceType::medium_tag ==
        specfem::element::medium_tag::acoustic) ||
       (PointSourceType::medium_tag == specfem::element::medium_tag::elastic)),
      "PointSourceType must be an acoustic or elastic point source");

#ifndef NDEBUG
  // Checks if the spectral element index is out of bounds

  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  // Checks if the spectral element has a source associated with it
  if (sources.source_domain_index_mapping(index.ispec) < 0) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.medium_types(index.ispec) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.wavefield_types(index.ispec) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  IndexType lcoord = index;
  lcoord.ispec = sources.source_domain_index_mapping(index.ispec);

#define SOURCE_MEDIUM_LOAD_ON_DEVICE(DIMENSION_TAG, MEDIUM_TAG)                \
  if constexpr (GET_TAG(DIMENSION_TAG) == specfem::dimension::type::dim2) {    \
    if constexpr (GET_TAG(MEDIUM_TAG) == PointSourceType::medium_tag) {        \
      sources                                                                  \
          .CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),               \
                                GET_NAME(MEDIUM_TAG))                          \
          .load_on_device(sources.timestep, lcoord, point_source);             \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      SOURCE_MEDIUM_LOAD_ON_DEVICE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef SOURCE_MEDIUM_LOAD_ON_DEVICE

  return;
}

/**
 * @brief Load source information on host at the given index
 *
 * Loads source information on device at the given index. Make sure you set
 the
 * correct timestep using `sources.update_timestep` before calling this
 * function.
 *
 * @tparam IndexType Point index type @ref specfem::point::index
 * @tparam PointSourceType Point source type @ref specfem::point::source
 * @param index Spectral element index to load source information
 * @param sources Source information for the domain
 * @param point_source Point source object to load source information into
 */
template <typename IndexType, typename PointSourceType>
void load_on_host(const IndexType index,
                  const specfem::compute::sources &sources,
                  PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      PointSourceType::is_point_source,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension == specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

  static_assert(
      ((PointSourceType::medium_tag ==
        specfem::element::medium_tag::acoustic) ||
       (PointSourceType::medium_tag == specfem::element::medium_tag::elastic)),
      "PointSourceType must be an acoustic or elastic point source");

#ifndef NDEBUG
  // Checks if the spectral element index is out of bounds
  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  // Checks if the spectral element has a source associated with it
  if (sources.h_source_domain_index_mapping(index.ispec) < 0) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.h_medium_types(index.ispec) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.h_wavefield_types(index.ispec) !=
      PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  IndexType lcoord = index;
  lcoord.ispec = sources.h_source_domain_index_mapping(index.ispec);

#define SOURCE_MEDIUM_LOAD_ON_HOST(DIMENSION_TAG, MEDIUM_TAG)                  \
  if constexpr (GET_TAG(DIMENSION_TAG) == specfem::dimension::type::dim2) {    \
    if constexpr (GET_TAG(MEDIUM_TAG) == PointSourceType::medium_tag) {        \
      sources                                                                  \
          .CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),               \
                                GET_NAME(MEDIUM_TAG))                          \
          .load_on_host(sources.timestep, lcoord, point_source);               \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      SOURCE_MEDIUM_LOAD_ON_HOST,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef SOURCE_MEDIUM_LOAD_ON_HOST

  return;
}

/**
 * @brief Store source information on device at the given index
 *
 * Loads source information on device at the given index. Make sure you set
 the
 * correct timestep using `sources.update_timestep` before calling this
 * function.
 *
 * @tparam IndexType Point index type @ref specfem::point::index
 * @tparam PointSourceType Point source type @ref specfem::point::source
 * @param index Spectral element index to load source information
 * @param point_source Point source object to load source information into
 * @param sources Source information for the domain
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void
store_on_device(const IndexType index, const PointSourceType &point_source,
                const specfem::compute::sources &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      PointSourceType::is_point_source,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension == specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

  static_assert(
      ((PointSourceType::medium_tag ==
        specfem::element::medium_tag::acoustic) ||
       (PointSourceType::medium_tag == specfem::element::medium_tag::elastic)),
      "PointSourceType must be an acoustic or elastic point source");

#ifndef NDEBUG
  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.source_domain_index_mapping(index.ispec) < 0) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.medium_types(index.ispec) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.wavefield_types(index.ispec) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  IndexType lcoord = index;
  lcoord.ispec = sources.source_domain_index_mapping(index.ispec);

#define SOURCE_MEDIUM_STORE_ON_DEVICE(DIMENSION_TAG, MEDIUM_TAG)               \
  if constexpr (GET_TAG(DIMENSION_TAG) == specfem::dimension::type::dim2) {    \
    if constexpr (GET_TAG(MEDIUM_TAG) == PointSourceType::medium_tag) {        \
      sources                                                                  \
          .CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),               \
                                GET_NAME(MEDIUM_TAG))                          \
          .store_on_device(sources.timestep, lcoord, point_source);            \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      SOURCE_MEDIUM_STORE_ON_DEVICE,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef SOURCE_MEDIUM_STORE_ON_DEVICE

  return;
}

/**
 * @brief Store source information on host at the given index
 *
 * Loads source information on device at the given index. Make sure you set
 the
 * correct timestep using `sources.update_timestep` before calling this
 * function.
 *
 * @tparam IndexType Point index type @ref specfem::point::index
 * @tparam PointSourceType Point source type @ref specfem::point::source
 * @param index Spectral element index to load source information
 * @param point_source Point source object to load source information into
 * @param sources Source information for the domain
 */
template <typename IndexType, typename PointSourceType>
void store_on_host(const IndexType index, const PointSourceType &point_source,
                   const specfem::compute::sources &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      PointSourceType::is_point_source,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension == specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

  static_assert(
      ((PointSourceType::medium_tag ==
        specfem::element::medium_tag::acoustic) ||
       (PointSourceType::medium_tag == specfem::element::medium_tag::elastic)),
      "PointSourceType must be an acoustic or elastic point source");

#ifndef NDEBUG
  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.h_source_domain_index_mapping(index.ispec) < 0) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.h_medium_types(index.ispec) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.h_wavefield_types(index.ispec) !=
      PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  IndexType lcoord = index;
  lcoord.ispec = sources.h_source_domain_index_mapping(index.ispec);

#define SOURCE_MEDIUM_STORE_ON_HOST(DIMENSION_TAG, MEDIUM_TAG)                 \
  if constexpr (GET_TAG(DIMENSION_TAG) == specfem::dimension::type::dim2) {    \
    if constexpr (GET_TAG(MEDIUM_TAG) == PointSourceType::medium_tag) {        \
      sources                                                                  \
          .CREATE_VARIABLE_NAME(source, GET_NAME(DIMENSION_TAG),               \
                                GET_NAME(MEDIUM_TAG))                          \
          .store_on_host(sources.timestep, lcoord, point_source);              \
    }                                                                          \
  }

  CALL_MACRO_FOR_ALL_MEDIUM_TAGS(
      SOURCE_MEDIUM_STORE_ON_HOST,
      WHERE(DIMENSION_TAG_DIM2) WHERE(MEDIUM_TAG_ELASTIC, MEDIUM_TAG_ACOUSTIC))

#undef SOURCE_MEDIUM_STORE_ON_HOST

  return;
}

} // namespace compute
} // namespace specfem
