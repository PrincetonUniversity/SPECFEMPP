#pragma once
#include "../impl/dim2/source_medium.tpp"
#include "../impl/source_medium.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/source.hpp"

#include <Kokkos_Core.hpp>

namespace specfem::assembly {
/**
 * @brief 2D template specialization for assembly-level source management
 *
 * This class manages sources within 2D assembled finite element meshes,
 * providing efficient data access patterns for spectral element simulations.
 * Sources are organized by medium type (elastic_psv, elastic_sh, acoustic,
 * poroelastic) and support time-dependent computations through source time
 * functions.
 *
 * Key features:
 * - Device/host memory management with Kokkos
 * - Medium-specific source organization
 * - Time-dependent source function evaluation
 * - Lagrange interpolant computation for spectral elements
 */
template <> struct sources<specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension
                                      ///< of the
                                      ///< mesh

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
   * @param jacobian_matrix Jacobian matrix for every quadrature point
   * @param properties Material properties for every quadrature point
   * @param t0 Initial time
   * @param dt Time step
   * @param nsteps Number of time steps
   */
  sources(
      std::vector<std::shared_ptr<specfem::sources::source<dimension_tag> > >
          &sources,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);
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
  std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
  get_sources_on_host(
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
  std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
  get_sources_on_device(
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
  IndexViewType element_indices;     ///< Mapping for every source to
                                     ///< local index within source_medium
  IndexViewType::HostMirror h_element_indices; ///< Host mirror of
                                               ///< domain_source_index_mapping
  IndexViewType source_indices;                ///< Mapping for every source to
                                ///< local index within source_medium
  IndexViewType::HostMirror h_source_indices; ///< Host mirror of
                                              ///< h_source_indices
  MediumTagViewType medium_types; ///< Medium type for every spectral element
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

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T)),
                      DECLARE(((specfem::assembly::sources_impl::source_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               source)))

  int timestep; ///< Current time step

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      DECLARE((IndexViewType, element_indices_forward),
              (IndexViewType::HostMirror, h_element_indices_forward),
              (IndexViewType, element_indices_backward),
              (IndexViewType::HostMirror, h_element_indices_backward),
              (IndexViewType, element_indices_adjoint),
              (IndexViewType::HostMirror, h_element_indices_adjoint),
              (IndexViewType, source_indices_forward),
              (IndexViewType::HostMirror, h_source_indices_forward),
              (IndexViewType, source_indices_backward),
              (IndexViewType::HostMirror, h_source_indices_backward),
              (IndexViewType, source_indices_adjoint),
              (IndexViewType::HostMirror, h_source_indices_adjoint)))

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void load_on_device(
      const IndexType index,
      const specfem::assembly::sources<specfem::dimension::type::dim2> &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend void load_on_host(
      const IndexType index,
      const specfem::assembly::sources<specfem::dimension::type::dim2> &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void store_on_device(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<specfem::dimension::type::dim2>
          &sources);

  template <typename IndexType, typename PointSourceType>
  friend void
  store_on_host(const IndexType index, const PointSourceType &point_source,
                const specfem::assembly::sources<specfem::dimension::type::dim2>
                    &sources);
};

/**
 * @defgroup SourceDataAccess2D Source Data Access Functions
 * @brief Data access functions for source operations
 */

/**
 * @brief Load source information on device at the given index
 * @ingroup SourceDataAccess2D
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
KOKKOS_INLINE_FUNCTION void load_on_device(
    const IndexType index,
    const specfem::assembly::sources<specfem::dimension::type::dim2> &sources,
    PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

#ifndef NDEBUG

  const int isource = index.imap;

  // Checks if the spectral element index is out of bounds

  if (index.ispec >= sources.nspec) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }
  if (sources.medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ == specfem::dimension::type::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Load source information on host at the given index
 * @ingroup SourceDataAccess2D
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
void load_on_host(
    const IndexType index,
    const specfem::assembly::sources<specfem::dimension::type::dim2> &sources,
    PointSourceType &point_source) {

  // Get the mapping from the iterator index

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

#ifndef NDEBUG
  const int isource = index.imap;

  // Checks if the spectral element index is out of bounds
  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.h_medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.h_wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ == specfem::dimension::type::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store source information on device at the given index
 * @ingroup SourceDataAccess2D
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
KOKKOS_INLINE_FUNCTION void store_on_device(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::dimension::type::dim2> &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

#ifndef NDEBUG
  const int isource = index.imap;

  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ == specfem::dimension::type::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store source information on host at the given index
 * @ingroup SourceDataAccess2D
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
void store_on_host(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::dimension::type::dim2> &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::dimension::type::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag == specfem::dimension::type::dim2,
                "IndexType must be a 2D index type");

#ifndef NDEBUG
  const int isource = index.imap;

  if ((index.ispec < 0) || (sources.nspec <= index.ispec)) {
    Kokkos::abort("Invalid spectral element index detected in source");
  }

  if (sources.h_medium_types(isource) != PointSourceType::medium_tag) {
    Kokkos::abort("Invalid medium detected in source");
  }

  if (sources.h_wavefield_types(isource) != PointSourceType::wavefield_tag) {
    Kokkos::abort("Invalid wavefield type detected in source");
  }
#endif

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ == specfem::dimension::type::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

} // namespace specfem::assembly
