#pragma once
#include "../impl/dim2/source_medium_data_access.tpp"

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/source.hpp"

#include <Kokkos_Core.hpp>

namespace specfem::assembly {
// clang-format off
/**
 * @brief 2D template specialization for assembly-level source management
 *
 * This class manages seismic sources within 2D assembled finite element meshes,
 * providing efficient data access patterns and organization for spectral
 * element wave propagation simulations. The class handles various source types
 * including force sources, moment tensor sources, and external sources across
 * different medium types.
 *
 * **Usage Examples:**
 *
 * @code{.cpp}
 * // 1. Initialize source assembly from configuration
 * std::vector<std::shared_ptr<specfem::sources::source<specfem::element::dimension_tag::dim2>>>
 * sources;
 * // ... populate sources from input files
 *
 * auto source_assembly =
 * specfem::assembly::sources<specfem::element::dimension_tag::dim2>( sources, mesh,
 * jacobian, element_types, 0.0, 0.01, 1000);
 *
 * // 2. Filter sources by criteria
 * auto [host_elements, host_sources] = source_assembly.get_sources_on_host(
 *     specfem::element::medium_tag::elastic_psv,
 *     specfem::element::property_tag::isotropic,
 *     specfem::element::boundary_tag::none,
 *     specfem::simulation::field_type::forward);
 *
 * // 3. Time integration with source updates
 * for (int step = 0; step < nsteps; ++step) {
 *     source_assembly.update_timestep(step);
 *
 *     // Device kernel example
 *     Kokkos::parallel_for("apply_sources",
 *         Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nsources),
 *         KOKKOS_LAMBDA(const int isource) {
 *             specfem::point::mapped_index<specfem::element::dimension_tag::dim2> index;
 *             specfem::point::source<...> point_source;
 *             load_on_device(index, source_assembly, point_source);
 *             // ... use point_source for assembly
 *         });
 * }
 * @endcode
 *
 */
// clang-format on
template <> struct sources<specfem::element::dimension_tag::dim2> {

public:
  /**
   * @name Public Constants
   */
  ///@{
  /**
   * @brief Dimension tag indicating this is a 2D implementation
   */
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim2;
  ///@}

private:
  /**
   * @name Private Type Definitions
   *
   * Kokkos view types used for efficient memory management and data access
   * patterns in source computations.
   */

  ///@{

  /**
   * @brief Kokkos view type for storing integer indices on device
   *
   * Used for various index mapping operations including source-to-element
   * mappings and element-to-source mappings.
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing medium classification tags on device
   *
   * Stores medium type information (elastic, acoustic, poroelastic) for each
   * source, enabling efficient medium-specific source processing.
   */
  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing wavefield simulation tags on device
   *
   * Identifies whether sources apply to forward, backward, or adjoint
   * wavefields, crucial for proper handling of different simulation types.
   */
  using WavefieldTagViewType = Kokkos::View<specfem::simulation::field_type *,
                                            Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing boundary condition tags on device
   *
   * Tracks boundary condition types associated with source locations, affecting
   * how sources are applied near domain boundaries.
   */
  using BoundaryTagViewType = Kokkos::View<specfem::element::boundary_tag *,
                                           Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing material property tags on device
   *
   * Stores material property classifications (isotropic, anisotropic, etc.) for
   * elements containing sources, enabling appropriate physics implementations.
   */
  using PropertyTagViewType = Kokkos::View<specfem::element::property_tag *,
                                           Kokkos::DefaultExecutionSpace>;

  ///@}

public:
  /**
   * @name Constructors and Destructors
   *
   * Object lifecycle management for source assembly data structures.
   */
  ///@{

  /**
   * @brief Default constructor
   *
   * Creates an empty source assembly with no initialized data. Use the
   * parameterized constructor to properly initialize with mesh and source data.
   */
  sources() = default;

  /**
   * @brief Construct source assembly from mesh and source configuration
   *
   * Initializes the complete source assembly data structure by processing
   * source definitions against the finite element mesh. This constructor:
   * - Maps sources to their containing spectral elements
   * - Classifies elements by medium, property, and boundary types
   * - Sets up efficient index mappings for source processing
   * - Initializes time-dependent source computations
   *
   * @param sources Vector of source objects read from input configuration,
   *                each containing position, time function, and type
   * information
   * @param mesh Finite element mesh providing element connectivity and geometry
   * @param jacobian_matrix Jacobian transformation matrices for all quadrature
   *                        points, enabling coordinate transformations
   * @param element_types Classification of elements by medium and property
   * types, determining appropriate physics implementations
   * @param t0 Initial simulation time (typically 0.0)
   * @param dt Time step size for temporal discretization
   * @param nsteps Total number of time steps in the simulation
   *
   * @code
   * // Initialize source assembly
   * auto source_assembly =
   * specfem::assembly::sources<specfem::element::dimension_tag::dim2>(
   * source_vector, mesh, jacobian, element_types, 0.0, 0.01, 1000);
   *
   * // Use in simulation
   * source_assembly.update_timestep(current_step);
   * @endcode
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
   * @brief Retrieve source indices for specified criteria on host memory
   *
   * Filters sources based on medium type, material properties, boundary
   * conditions, and wavefield application, returning host-accessible views
   * for CPU-based processing.
   *
   * @param medium Physical medium type (elastic, acoustic, poroelastic)
   *               determining wave equation formulation
   * @param property Material property classification (isotropic, anisotropic)
   *                 affecting constitutive relationships
   * @param boundary Boundary condition type (free surface, absorbing, etc.)
   *                 influencing source application near boundaries
   * @param wavefield Simulation type (forward, adjoint, backward) for proper
   *                  source handling in different computational modes
   *
   * @return std::tuple containing:
   *         - Element indices view: Maps to spectral elements containing
   * sources
   *         - Source indices view: Maps to sources within the filtered set
   *
   * @code
   * // Get elastic sources for forward simulation
   * auto [elements, sources] = source_assembly.get_sources_on_host(
   *     specfem::element::medium_tag::elastic,
   *     specfem::element::property_tag::isotropic,
   *     specfem::element::boundary_tag::none,
   *     specfem::simulation::field_type::forward);
   * @endcode
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> >
  get_sources_on_host(const specfem::element::medium_tag medium,
                      const specfem::element::property_tag property,
                      const specfem::element::boundary_tag boundary,
                      const specfem::simulation::field_type wavefield) const;

  /**
   * @brief Retrieve source indices for specified criteria on device memory
   *
   * Device-accessible version of source filtering for GPU-based computations.
   * Returns Kokkos views suitable for use in device kernels, enabling efficient
   * parallel source processing during assembly operations.
   *
   * @param medium Physical medium type determining appropriate wave equations
   * @param property Material property type affecting physics implementation
   * @param boundary Boundary condition classification for proper source
   * handling
   * @param wavefield Simulation mode for correct source application
   *
   * @return std::tuple containing device-accessible views:
   *         - Element indices: Spectral elements containing filtered sources
   *         - Source indices: Sources matching the specified criteria
   *
   * @code
   * // Get device views for acoustic sources in adjoint simulation
   * auto [dev_elements, dev_sources] = source_assembly.get_sources_on_device(
   *     specfem::element::medium_tag::acoustic,
   *     specfem::element::property_tag::isotropic,
   *     specfem::element::boundary_tag::none,
   *     specfem::simulation::field_type::adjoint);
   * @endcode
   */
  std::tuple<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>,
             Kokkos::View<int *, Kokkos::DefaultExecutionSpace> >
  get_sources_on_device(const specfem::element::medium_tag medium,
                        const specfem::element::property_tag property,
                        const specfem::element::boundary_tag boundary,
                        const specfem::simulation::field_type wavefield) const;

  /**
   * @brief Update the current simulation time step
   *
   * Sets the active time step index for time-dependent source function
   * evaluation. This method must be called before each time step to ensure
   * source time functions are evaluated at the correct temporal point.
   * The time step is used by `load_on_device` and `load_on_host` methods
   * for accessing time-dependent source amplitudes.
   *
   * @param timestep Current time step index (0-based, typically from main
   *                 time integration loop)
   *
   * @code
   * // Time integration loop
   * for (int step = 0; step < nsteps; ++step) {
   *     source_assembly.update_timestep(step);
   *     // ... perform assembly operations
   * }
   * @endcode
   */
  void update_timestep(const int timestep) { this->timestep = timestep; }

private:
  /**
   * @brief Total number of spectral elements in the computational domain
   *
   */
  int nspec;

  /**
   * @brief Device view mapping spectral elements to local source medium indices
   *
   * For each spectral element containing a source, provides the corresponding
   * local index within the specialized source medium data structures. This
   * enables efficient source-to-element lookup during assembly.
   *
   */
  IndexViewType source_domain_index_mapping;

  /**
   * @brief Host mirror of source domain index mapping
   *
   * Host-accessible copy of `source_domain_index_mapping` for initialization
   * and debugging operations.
   */
  IndexViewType::host_mirror_type h_source_domain_index_mapping;

  /**
   * @brief Device view mapping sources to element indices within source medium
   *
   * Maps each source to its corresponding local index within the source medium
   * data structures, enabling efficient source processing.
   */
  IndexViewType element_indices;

  /**
   * @brief Host mirror of element indices mapping
   *
   * Host-accessible copy of `element_indices` for setup and validation.
   */
  IndexViewType::host_mirror_type h_element_indices;

  /**
   * @brief Device view mapping sources to local source medium indices
   *
   * Provides the local index within source medium data structures for each
   * source, facilitating direct source access during computation.
   */
  IndexViewType source_indices;

  /**
   * @brief Host mirror of source indices mapping
   *
   * Host-accessible copy of `source_indices` for configuration management.
   */
  IndexViewType::host_mirror_type h_source_indices;

  /**
   * @brief Device view storing medium types for each spectral element
   *
   * Classifies the physical medium (elastic, acoustic, poroelastic) for
   * elements in the source domain, determining appropriate physics models.
   */
  MediumTagViewType medium_types;

  /**
   * @brief Host mirror of medium types
   *
   * Host-accessible copy of `medium_types` for initialization and analysis.
   */
  MediumTagViewType::host_mirror_type h_medium_types;

  /**
   * @brief Device view storing wavefield types for source applications
   *
   * Identifies whether sources apply to forward, backward, or adjoint
   * wavefields, critical for proper simulation mode handling.
   */
  WavefieldTagViewType wavefield_types;

  /**
   * @brief Host mirror of wavefield types
   *
   * Host-accessible copy of `wavefield_types` for setup operations.
   */
  WavefieldTagViewType::host_mirror_type h_wavefield_types;

  /**
   * @brief Device view storing boundary condition types for spectral elements
   *
   * Tracks boundary condition classifications at source locations, affecting
   * how sources are applied near domain boundaries and material interfaces.
   */
  BoundaryTagViewType boundary_types;

  /**
   * @brief Host mirror of boundary condition types
   *
   * Host-accessible copy of `boundary_types` for configuration management.
   */
  BoundaryTagViewType::host_mirror_type h_boundary_types;

  /**
   * @brief Device view storing material property types for spectral elements
   *
   * Stores material property classifications (isotropic, anisotropic, etc.)
   * for elements in the source domain, enabling appropriate constitutive
   * models.
   */
  PropertyTagViewType property_types;

  /**
   * @brief Host mirror of material property types
   *
   * Host-accessible copy of `property_types` for setup and validation.
   */
  PropertyTagViewType::host_mirror_type h_property_types;
  ///@}

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                  POROELASTIC, ELASTIC_PSV_T)),
                      DECLARE(((specfem::assembly::sources_impl::source_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               source)))

  /**
   * @brief Current simulation time step
   *
   */
  int timestep;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT),
       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                    COMPOSITE_STACEY_DIRICHLET)),
      DECLARE((IndexViewType, element_indices_forward),
              (IndexViewType::host_mirror_type, h_element_indices_forward),
              (IndexViewType, element_indices_backward),
              (IndexViewType::host_mirror_type, h_element_indices_backward),
              (IndexViewType, element_indices_adjoint),
              (IndexViewType::host_mirror_type, h_element_indices_adjoint),
              (IndexViewType, source_indices_forward),
              (IndexViewType::host_mirror_type, h_source_indices_forward),
              (IndexViewType, source_indices_backward),
              (IndexViewType::host_mirror_type, h_source_indices_backward),
              (IndexViewType, source_indices_adjoint),
              (IndexViewType::host_mirror_type, h_source_indices_adjoint)))

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void load_on_device(
      const IndexType index,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
          &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend void load_on_host(
      const IndexType index,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
          &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void store_on_device(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
          &sources);

  template <typename IndexType, typename PointSourceType>
  friend void store_on_host(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
          &sources);
};

/**
 * @defgroup SourceDataAccess Source Data Access Functions
 */

/**
 * @brief Load source data for device-based computations
 * @ingroup SourceDataAccess
 *
 * Load source information into point source structures for use in device
 *
 * @tparam IndexType 2D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 2D point source type matching medium and wavefield
 tags
 * @param index Spectral element index containing source location information
 * @param sources Source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @note In debug builds, the function performs validity checks on indices and
 * medium/wavefield type consistency.
 *
 * @code
 * // Usage in device kernel
 * specfem::point::mapped_index<specfem::element::dimension_tag::dim2> idx;
 * specfem::point::source<...> src;
 * load_on_device(idx, source_assembly, src);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void load_on_device(
    const IndexType index,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
        &sources,
    PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
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
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Load source data for host-based computations
 * @ingroup SourceDataAccess
 *
 * Load source information into point source structures for use in host
 *
 * @tparam IndexType 2D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 2D point source type matching medium and wavefield
 tags
 * @param index Spectral element index containing source location information
 * @param sources Source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @note In debug builds, the function performs validity checks on indices and
 * medium/wavefield type consistency.
 *
 * @code
 * // Usage in host code
 * specfem::point::mapped_index<specfem::element::dimension_tag::dim2> idx;
 * specfem::point::source<...> src;
 * load_on_host(idx, source_assembly, src);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
void load_on_host(
    const IndexType index,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
        &sources,
    PointSourceType &point_source) {

  // Get the mapping from the iterator index

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
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
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store source data from device computations
 * @ingroup SourceDataAccess
 *
 * GPU-optimized version for writing computed source data back to the assembly
 * structure. Used in device kernels where source parameters are updated.
 *
 * @tparam IndexType 2D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 2D point source type matching medium and wavefield
 * tags
 * @param index Spectral element index identifying storage location
 * @param point_source [in] Source data to be stored in assembly
 * @param sources Source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @note In debug builds, the function performs validity checks on indices and
 * medium/wavefield type consistency.
 *
 * @code
 * // Usage in device kernel for source modification
 * specfem::point::source<...> modified_src;
 * // ... modify source parameters
 * store_on_device(idx, modified_src, source_assembly);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void store_on_device(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
        &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
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
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store source data from host computations
 * @ingroup SourceDataAccess
 *
 * Host-optimized version for writing computed source data back to the
 * assembly structure. Used in CPU code where source parameters are updated.
 *
 * @tparam IndexType 2D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 2D point source type matching medium and wavefield
 * tags
 * @param index Spectral element index identifying storage location
 * @param point_source [in] Source data to be stored in assembly
 * @param sources Source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @note In debug builds, the function performs validity checks on indices and
 * medium/wavefield type consistency.
 *
 * @code
 * // Usage in host code for source modification
 * specfem::point::source<...> modified_src;
 * // ... modify source parameters
 * store_on_host(idx, modified_src, source_assembly);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
void store_on_host(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim2>
        &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
                "PointSourceType must be a 2D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim2,
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
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim2) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

} // namespace specfem::assembly
