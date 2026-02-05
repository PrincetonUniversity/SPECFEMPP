#pragma once
#include "../impl/dim3/source_medium.tpp"
#include "../impl/source_medium.hpp"
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
 * @brief 3D template specialization for assembly-level source management
 *
 * This class manages seismic sources within 3D assembled finite element meshes,
 * providing efficient data access patterns and organization for spectral
 * element wave propagation simulations. The class handles various source types
 * including force sources, moment tensor sources, and external sources across
 * different medium types.
  *
 * **Usage Examples:**
 *
 * @code
 * // 1. Initialize source assembly from configuration
 * std::vector<std::shared_ptr<specfem::sources::source<specfem::element::dimension_tag::dim3>>> sources;
 * // ... populate sources from input files
 *
 * auto source_assembly = specfem::assembly::sources<specfem::element::dimension_tag::dim3>(
 *     sources, mesh, jacobian, element_types, 0.0, 0.01, 1000);
 *
 * // 2. Filter sources by criteria
 * auto [host_elements, host_sources] = source_assembly.get_sources_on_host(
 *     specfem::element::medium_tag::elastic,
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
 *             specfem::point::mapped_index<specfem::element::dimension_tag::dim3> index;
 *             specfem::point::source<...> point_source;
 *             load_on_device(index, source_assembly, point_source);
 *             // ... use point_source for assembly
 *         });
 * }
 * @endcode
 *
 */
// clang-format on
template <> struct sources<specfem::element::dimension_tag::dim3> {

public:
  /**
   * @name Public Constants
   */
  ///@{
  /**
   * @brief Dimension tag indicating this is a 3D implementation
   */
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  ///@}

private:
  /**
   * @name Private Type Definitions
   *
   * Kokkos view types used for efficient memory management and data access
   * patterns in 3D source computations.
   */

  ///@{

  /**
   * @brief Kokkos view type for storing integer indices on device
   *
   * Used for various index mapping operations including source-to-element
   * mappings and element-to-source mappings in 3D domains.
   */
  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing medium classification tags on device
   *
   * Stores medium type information (elastic) for each source in 3D,
   * enabling efficient medium-specific source processing.
   */
  using MediumTagViewType = Kokkos::View<specfem::element::medium_tag *,
                                         Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing wavefield simulation tags on device
   *
   * Identifies whether sources apply to forward, backward, or adjoint
   * wavefields, crucial for proper handling of different simulation types in
   * 3D.
   */
  using WavefieldTagViewType = Kokkos::View<specfem::simulation::field_type *,
                                            Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing boundary condition tags on device
   *
   * Tracks boundary condition types associated with source locations, affecting
   * how sources are applied near domain boundaries in 3D meshes.
   */
  using BoundaryTagViewType = Kokkos::View<specfem::element::boundary_tag *,
                                           Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for storing material property tags on device
   *
   * Stores material property classifications (isotropic) for elements
   * containing sources, enabling appropriate physics implementations in 3D.
   */
  using PropertyTagViewType = Kokkos::View<specfem::element::property_tag *,
                                           Kokkos::DefaultExecutionSpace>;

  ///@}

public:
  /**
   * @name Constructors and Destructors
   *
   * Object lifecycle management for 3D source assembly data structures.
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
   * @brief Construct 3D source assembly from mesh and source configuration
   *
   * Initializes the complete 3D source assembly data structure by processing
   * source definitions against the finite element mesh. This constructor:
   * - Maps sources to their containing spectral elements in 3D space
   * - Classifies elements by medium, property, and boundary types
   * - Sets up efficient index mappings for 3D source processing
   * - Initializes time-dependent source computations
   *
   * @param sources Vector of 3D source objects read from input configuration,
   *                each containing position, time function, and type
   * information
   * @param mesh 3D finite element mesh providing element connectivity and
   * geometry
   * @param jacobian_matrix Jacobian transformation matrices for all quadrature
   *                        points in 3D, enabling coordinate transformations
   * @param element_types Classification of 3D elements by medium and property
   * types, determining appropriate physics implementations
   * @param t0 Initial simulation time (typically 0.0)
   * @param dt Time step size for temporal discretization
   * @param nsteps Total number of time steps in the simulation
   *
   * @code
   * // Initialize 3D source assembly
   * auto source_assembly =
   * specfem::assembly::sources<specfem::element::dimension_tag::dim3>(
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
   * @name Source Query Methods
   *
   * Methods for retrieving filtered 3D source information based on physical
   * and computational criteria.
   */
  ///@{

  /**
   * @brief Retrieve 3D source indices for specified criteria on host memory
   *
   * Filters sources based on medium type, material properties, boundary
   * conditions, and wavefield application, returning host-accessible views
   * for CPU-based processing in 3D domains.
   *
   * @param medium Physical medium type (elastic) determining wave equation
   *               formulation for 3D problems
   * @param property Material property classification (isotropic) affecting
   *                 constitutive relationships in 3D
   * @param boundary Boundary condition type (none) influencing source
   *                 application near 3D domain boundaries
   * @param wavefield Simulation type (forward, adjoint, backward) for proper
   *                  source handling in different computational modes
   *
   * @return std::tuple containing:
   *         - Element indices view: Maps to 3D spectral elements containing
   * sources
   *         - Source indices view: Maps to sources within the filtered set
   *
   * @code
   * // Get elastic sources for forward simulation in 3D
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
   * @brief Retrieve 3D source indices for specified criteria on device memory
   *
   * Device-accessible version of source filtering for GPU-based computations
   * in 3D domains. Returns Kokkos views suitable for use in device kernels,
   * enabling efficient parallel source processing during assembly operations.
   *
   * @param medium Physical medium type determining appropriate wave equations
   *               for 3D elastic problems
   * @param property Material property type affecting physics implementation
   * @param boundary Boundary condition classification for proper source
   * handling
   * @param wavefield Simulation mode for correct source application
   *
   * @return std::tuple containing device-accessible views:
   *         - Element indices: 3D spectral elements containing filtered sources
   *         - Source indices: Sources matching the specified criteria
   *
   * @code
   * // Get device views for elastic sources in adjoint 3D simulation
   * auto [dev_elements, dev_sources] = source_assembly.get_sources_on_device(
   *     specfem::element::medium_tag::elastic,
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
  ///@}

  /**
   * @name Time Management
   *
   * Methods for controlling temporal aspects of 3D source computations.
   */
  ///@{

  /**
   * @brief Update the current simulation time step
   *
   * Sets the active time step index for time-dependent source function
   * evaluation in 3D simulations. This method must be called before each
   * time step to ensure source time functions are evaluated at the correct
   * temporal point. The time step is used by `load_on_device` and
   * `load_on_host` methods for accessing time-dependent source amplitudes.
   *
   * @param timestep Current time step index (0-based, typically from main
   *                 time integration loop)
   *
   * @code
   * // Time integration loop for 3D simulation
   * for (int step = 0; step < nsteps; ++step) {
   *     source_assembly.update_timestep(step);
   *     // ... perform 3D assembly operations
   * }
   * @endcode
   */
  void update_timestep(const int timestep) { this->timestep = timestep; }
  ///@}

private:
  /**
   * @brief Total number of spectral elements in the 3D computational domain
   *
   * Used for dimensioning index mapping arrays and validating element indices
   * during 3D source assembly operations.
   */
  int nspec;

  /**
   * @brief Device view mapping 3D spectral elements to local source medium
   * indices
   *
   * For each 3D spectral element containing a source, provides the
   * corresponding local index within the specialized source medium data
   * structures. This enables efficient source-to-element lookup during
   * assembly.
   */
  IndexViewType source_domain_index_mapping;

  /**
   * @brief Host mirror of 3D source domain index mapping
   *
   * Host-accessible copy of `source_domain_index_mapping` for initialization
   * and debugging operations in 3D.
   */
  IndexViewType::HostMirror h_source_domain_index_mapping;

  /**
   * @brief Device view mapping sources to element indices within 3D source
   * medium
   *
   * Maps each source to its corresponding local index within the 3D source
   * medium data structures, enabling efficient source processing.
   */
  IndexViewType element_indices;

  /**
   * @brief Host mirror of 3D element indices mapping
   *
   * Host-accessible copy of `element_indices` for setup and validation.
   */
  IndexViewType::HostMirror h_element_indices;

  /**
   * @brief Device view mapping sources to local 3D source medium indices
   *
   * Provides the local index within 3D source medium data structures for each
   * source, facilitating direct source access during computation.
   */
  IndexViewType source_indices;

  /**
   * @brief Host mirror of 3D source indices mapping
   *
   * Host-accessible copy of `source_indices` for configuration management.
   */
  IndexViewType::HostMirror h_source_indices;

  /**
   * @brief Device view storing medium types for each 3D spectral element
   *
   * Classifies the physical medium (elastic) for elements in the 3D source
   * domain, determining appropriate physics models.
   */
  MediumTagViewType medium_types;

  /**
   * @brief Host mirror of 3D medium types
   *
   * Host-accessible copy of `medium_types` for initialization and analysis.
   */
  MediumTagViewType::HostMirror h_medium_types;

  /**
   * @brief Device view storing wavefield types for 3D source applications
   *
   * Identifies whether sources apply to forward, backward, or adjoint
   * wavefields, critical for proper simulation mode handling in 3D.
   */
  WavefieldTagViewType wavefield_types;

  /**
   * @brief Host mirror of 3D wavefield types
   *
   * Host-accessible copy of `wavefield_types` for setup operations.
   */
  WavefieldTagViewType::HostMirror h_wavefield_types;

  /**
   * @brief Device view storing boundary condition types for 3D spectral
   * elements
   *
   * Tracks boundary condition classifications at source locations, affecting
   * how sources are applied near 3D domain boundaries and material interfaces.
   */
  BoundaryTagViewType boundary_types;

  /**
   * @brief Host mirror of 3D boundary condition types
   *
   * Host-accessible copy of `boundary_types` for configuration management.
   */
  BoundaryTagViewType::HostMirror h_boundary_types;

  /**
   * @brief Device view storing material property types for 3D spectral elements
   *
   * Stores material property classifications (isotropic) for elements in the
   * 3D source domain, enabling appropriate constitutive models.
   */
  PropertyTagViewType property_types;

  /**
   * @brief Host mirror of 3D material property types
   *
   * Host-accessible copy of `property_types` for setup and validation.
   */
  PropertyTagViewType::HostMirror h_property_types;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
                      DECLARE(((specfem::assembly::sources_impl::source_medium,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               source)))

  /**
   * @brief Current simulation time step for 3D source evaluation
   *
   * Time step index used for evaluating time-dependent source functions
   * in 3D simulations. Updated via `update_timestep()` method.
   */
  int timestep;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC),
       PROPERTY_TAG(ISOTROPIC), BOUNDARY_TAG(NONE)),
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
      const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
          &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend void load_on_host(
      const IndexType index,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
          &sources,
      PointSourceType &point_source);

  template <typename IndexType, typename PointSourceType>
  friend KOKKOS_INLINE_FUNCTION void store_on_device(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
          &sources);

  template <typename IndexType, typename PointSourceType>
  friend void store_on_host(
      const IndexType index, const PointSourceType &point_source,
      const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
          &sources);
};

/**
 * @defgroup SourceDataAccess Source Data Access Functions
 */

/**
 * @brief Load 3D source data for device-based computations
 * @ingroup SourceDataAccess
 *
 * Efficiently retrieves 3D source information optimized for GPU kernels,
 * including time-dependent source functions, spatial coordinates, and
 * medium-specific properties. This function provides type-safe access with
 * compile-time validation for 3D elastic problems.
 *
 * @tparam IndexType 3D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 3D point source type matching medium and wavefield
 * tags
 * @param index 3D spectral element index containing source location information
 * @param sources 3D source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with 3D source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @warning Function will abort on invalid indices or mismatched
 * medium/wavefield types
 *
 * @code
 * // Usage in 3D device kernel
 * specfem::point::mapped_index<specfem::element::dimension_tag::dim3> idx;
 * specfem::point::source<...> src;
 * load_on_device(idx, source_assembly, src);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void load_on_device(
    const IndexType index,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
        &sources,
    PointSourceType &point_source) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when loading sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "PointSourceType must be a 3D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "IndexType must be a 3D index type");

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
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim3) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Load 3D source data for host-based computations
 * @ingroup SourceDataAccess
 *
 * CPU-optimized version of 3D source data retrieval for host-side processing,
 * setup operations, and analysis. Provides identical interface to device
 * version with host memory access patterns for 3D elastic problems.
 *
 * @tparam IndexType 3D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 3D point source type matching medium and wavefield
 * tags
 * @param index 3D spectral element index containing source location information
 * @param sources 3D source assembly with current timestep configuration
 * @param point_source [out] Output structure populated with 3D source data
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @warning Function will abort on invalid indices or mismatched
 * medium/wavefield types
 *
 * @code
 * // Usage in 3D host code
 * specfem::point::mapped_index<specfem::element::dimension_tag::dim3> idx;
 * specfem::point::source<...> src;
 * load_on_host(idx, source_assembly, src);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
void load_on_host(
    const IndexType index,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
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
                    specfem::element::dimension_tag::dim3,
                "PointSourceType must be a 3D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "IndexType must be a 3D index type");

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
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim3) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.load_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store 3D source data from device computations
 * @ingroup SourceDataAccess
 *
 * Efficiently writes computed 3D source data back to the assembly structure
 * during GPU-based operations. Used for adjoint computations, source
 * inversion, and iterative algorithms where 3D source terms are modified.
 *
 * @tparam IndexType 3D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 3D point source type matching medium and wavefield
 * tags
 * @param index 3D spectral element index identifying storage location
 * @param point_source [in] 3D source data to be stored in assembly
 * @param sources 3D source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @warning Function will abort on invalid indices or mismatched
 * medium/wavefield types
 *
 * @code
 * // Usage in 3D device kernel for adjoint computation
 * specfem::point::source<...> computed_src;
 * // ... compute adjoint source in 3D
 * store_on_device(idx, computed_src, source_assembly);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
KOKKOS_INLINE_FUNCTION void store_on_device(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
        &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "PointSourceType must be a 3D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "IndexType must be a 3D index type");

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
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim3) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_device(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

/**
 * @brief Store 3D source data from host computations
 * @ingroup SourceDataAccess3D
 *
 * CPU-optimized version for writing computed 3D source data back to the
 * assembly structure. Used in preprocessing, initialization, and host-based
 * iterative algorithms where 3D source parameters are updated.
 *
 * @tparam IndexType 3D point index type (non-SIMD) for element-source mapping
 * @tparam PointSourceType 3D point source type matching medium and wavefield
 * tags
 * @param index 3D spectral element index identifying storage location
 * @param point_source [in] 3D source data to be stored in assembly
 * @param sources 3D source assembly with current timestep configuration
 *
 * @pre Call `sources.update_timestep(step)` before using this function
 * @warning Function will abort on invalid indices or mismatched
 * medium/wavefield types
 *
 * @code
 * // Usage in 3D host code for source modification
 * specfem::point::source<...> modified_src;
 * // ... modify 3D source parameters
 * store_on_host(idx, modified_src, source_assembly);
 * @endcode
 */
template <typename IndexType, typename PointSourceType>
void store_on_host(
    const IndexType index, const PointSourceType &point_source,
    const specfem::assembly::sources<specfem::element::dimension_tag::dim3>
        &sources) {

  static_assert(IndexType::using_simd == false,
                "IndexType must not use SIMD when storing sources");

  static_assert(
      specfem::data_access::is_point<PointSourceType>::value &&
          specfem::data_access::is_source<PointSourceType>::value,
      "PointSourceType must be a point source type specfem::point::source");

  static_assert(PointSourceType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "PointSourceType must be a 3D point source type");

  static_assert(IndexType::dimension_tag ==
                    specfem::element::dimension_tag::dim3,
                "IndexType must be a 3D index type");

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
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
      CAPTURE((source, sources.source)) {
        if constexpr (_dimension_tag_ ==
                      specfem::element::dimension_tag::dim3) {
          if constexpr (_medium_tag_ == PointSourceType::medium_tag) {
            _source_.store_on_host(sources.timestep, index, point_source);
          }
        }
      })

  return;
}

} // namespace specfem::assembly
