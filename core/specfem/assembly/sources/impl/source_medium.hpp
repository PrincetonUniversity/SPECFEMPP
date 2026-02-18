#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

// Forward declarations
namespace specfem {
namespace algorithms {
template <specfem::element::dimension_tag DimensionTag>
specfem::point::local_coordinates<DimensionTag> locate_point(
    const specfem::point::global_coordinates<DimensionTag> &coordinates,
    const specfem::assembly::mesh<DimensionTag> &mesh);
}
} // namespace specfem

namespace specfem::assembly::sources_impl {

/**
 * @brief Template metaprogramming utility for multi-dimensional array type
 * generation
 *
 * Recursively builds pointer types for multi-dimensional arrays used in source
 * data storage. This helper enables compile-time generation of appropriate
 * Kokkos view types for different dimensional problems.
 *
 * @tparam T Base data type (typically `type_real`)
 * @tparam Rank Number of array dimensions to generate
 */
template <typename T, int Rank> struct ExtentImpl {
  /**
   * @brief Recursive type definition for multi-dimensional pointer
   */
  using type = typename ExtentImpl<T, Rank - 1>::type *;
};

/**
 * @brief Base case specialization for rank-0 arrays
 *
 * @tparam T Base data type
 */
template <typename T> struct ExtentImpl<T, 0> {
  /**
   * @brief Base type for rank-0 case
   */
  using type = T;
};

} // namespace specfem::assembly::sources_impl

namespace specfem::assembly::sources_impl {
/**
 * @brief Medium-specific source data management for spectral element
 * simulations
 *
 * This class manages source information for a specific medium type within the
 * spectral element framework, providing efficient storage and access patterns
 * for source time functions, Lagrange interpolants, and element mappings.
 * Sources are organized by medium type to enable optimized computational
 * kernels.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 * @tparam MediumTag Physical medium type (`elastic_psv`, `acoustic`, etc.)
 *
 * @note This class is an implementation detail and should be only used within
 * @ref specfem::assembly::sources.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct source_medium {

public:
  /**
   * @name Public Constants
   *
   * Template-dependent constants defining the characteristics of this
   * source medium specialization.
   */
  ///@{

  /**
   * @brief Medium type identifier for this specialization
   */
  constexpr static auto medium_tag = MediumTag;

  /**
   * @brief Spatial dimension identifier for this specialization
   */
  constexpr static auto dimension_tag = DimensionTag;

  /**
   * @brief Number of spatial dimensions (2 for 2D, 3 for 3D)
   */
  constexpr static auto ndim =
      (DimensionTag == specfem::element::dimension_tag::dim2) ? 2 : 3;

  /**
   * @brief Rank of source array (4 for 2D: [sources][components][z][x],
   *        5 for 3D: [sources][components][z][y][x])
   */
  constexpr static int source_array_rank = ndim + 2;
  ///@}

private:
  /**
   * @name Private Type Definitions
   *
   * Kokkos view types for efficient memory management and data access.
   */
  ///@{

  /**
   * @brief Device view type for storing integer indices
   */
  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Device view type for storing source time functions
   *
   * Layout: [nsources][ncomponents][nsteps] with right-aligned memory layout
   * for optimal performance.
   */
  using SourceTimeFunctionView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Device view type for storing multi-dimensional source arrays
   *
   * Uses ExtentImpl to generate appropriate multi-dimensional pointer types.
   * Layout varies by dimension:
   * - 2D: [nsources][ncomponents][ngllz][ngllx]
   * - 3D: [nsources][ncomponents][ngllz][nglly][ngllx]
   */
  using SourceArrayView =
      Kokkos::View<typename specfem::assembly::sources_impl::ExtentImpl<
                       type_real, source_array_rank>::type,
                   Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;
  ///@}

  /**
   * @brief Number of field components for this medium type
   *
   * Determined by medium-specific attributes (e.g., 2 for elastic_psv,
   * 1 for acoustic, 3 for elastic in 3D).
   */
  constexpr static int components =
      specfem::element::attributes<dimension_tag, MediumTag>::components;

public:
  /**
   * @name Constructors
   *
   * Object lifecycle management for medium-specific source data structures.
   */
  ///@{

  /**
   * @brief Default constructor
   *
   * Creates an empty source medium structure. Use the parameterized
   * constructor to initialize with actual source data.
   */
  source_medium() = default;

  /**
   * @brief Construct 2D source medium from mesh and source data
   *
   * Initializes source data structures for 2D problems by:
   * - Computing Lagrange interpolants at source locations
   * - Setting up time function storage and evaluation
   * - Creating efficient device/host memory layouts
   * - Establishing element-to-source mappings
   *
   * @param sources Vector of source objects for this medium type
   * @param mesh 2D finite element mesh with element connectivity
   * @param jacobian_matrix Jacobian matrices for coordinate transformations
   * @param element_types Element classification data
   * @param t0 Initial time for source evaluation
   * @param dt Time step size for discretization
   * @param nsteps Total number of time steps
   */
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<dimension_tag> > > &sources,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);

  /**
   * @brief Construct 3D source medium from mesh and source data
   *
   * Initializes source data structures for 3D problems by:
   * - Computing Lagrange interpolants at source locations in 3D space
   * - Setting up time function storage and evaluation
   * - Creating efficient device/host memory layouts for 3D arrays
   * - Establishing element-to-source mappings
   *
   * @param sources Vector of source objects for this medium type
   * @param mesh 3D finite element mesh with element connectivity
   * @param jacobian_matrix Jacobian matrices for 3D coordinate transformations
   * @param element_types Element classification data
   * @param t0 Initial time for source evaluation
   * @param dt Time step size for discretization
   * @param nsteps Total number of time steps
   */
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<dimension_tag> > > &sources,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);
  ///@}

  /**
   * @name Data Members
   *
   * Core data storage for medium-specific source information with
   * device/host memory pairs for efficient computation.
   */
  ///@{

  /**
   * @brief Device view mapping sources to spectral element indices
   *
   * Contains the spectral element index (`ispec`) for each source in this
   * medium, enabling direct lookup during assembly operations.
   */
  IndexView source_index_mapping;

  /**
   * @brief Host mirror of spectral element index mapping
   *
   * Host-accessible copy of `source_index_mapping` for initialization and
   * debugging.
   */
  IndexView::HostMirror h_source_index_mapping;

  /**
   * @brief Device view storing time-dependent source functions
   *
   * 3D array layout: [nsources][ncomponents][nsteps]
   * Contains pre-computed or interpolated source amplitudes for each time step,
   * component, and source in this medium.
   */
  SourceTimeFunctionView source_time_function;

  /**
   * @brief Host mirror of source time functions
   *
   * Host-accessible copy of `source_time_function` for setup and analysis.
   */
  SourceTimeFunctionView::HostMirror h_source_time_function;

  /**
   * @brief Device view storing Lagrange interpolation weights
   *
   * Multi-dimensional array containing pre-computed Lagrange interpolant values
   * at source locations within spectral elements:
   * - 2D: [nsources][ncomponents][ngllz][ngllx]
   * - 3D: [nsources][ncomponents][ngllz][nglly][ngllx]
   */
  SourceArrayView source_array;

  /**
   * @brief Host mirror of Lagrange interpolation weights
   *
   * Host-accessible copy of `source_array` for initialization and validation.
   */
  typename SourceArrayView::HostMirror h_source_array;
  ///@}

  /**
   * @name Data Access Functions
   *
   * Template functions for efficient source data loading and storing on
   * both device and host execution spaces, specialized for 2D and 3D problems.
   */
  ///@{

  /**
   * @brief Load 2D source data for device-based computations
   *
   * Retrieves source information from device memory for 2D problems, including
   * time-dependent amplitudes and Lagrange interpolation weights.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source function evaluation
   * @param index Element index containing source location information
   * @param point_source [out] Output structure populated with 2D source data
   *
   * @note This function is an implementation detail and is called by
   *       higher-level @c load_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  /**
   * @brief Load 3D source data for device-based computations
   *
   * Retrieves source information from device memory for 3D problems, including
   * time-dependent amplitudes and Lagrange interpolation weights.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source function evaluation
   * @param index Element index containing source location information
   * @param point_source [out] Output structure populated with 3D source data
   * @note This function is an implementation detail and is called by
   *       higher-level @c load_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  /**
   * @brief Store 2D source data from device computations
   *
   * Writes computed source data back to device memory for 2D problems.
   * Used in adjoint computations and source inversion algorithms where
   * source parameters are updated during GPU-based iterative processes.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source data storage
   * @param index Element index identifying source storage location
   * @param point_source [in] Source data to be stored in device memory
   * @note This function is an implementation detail and is called by
   *       higher-level @c store_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  /**
   * @brief Store 3D source data from device computations
   *
   * Writes computed source data back to device memory for 3D problems.
   * Used in adjoint computations and source inversion algorithms where
   * source parameters are updated during GPU-based iterative processes.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source data storage
   * @param index Element index identifying source storage location
   * @param point_source [in] Source data to be stored in device memory
   * @note This function is an implementation detail and is called by
   *       higher-level @c store_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  /**
   * @brief Load 2D source data for host-based computations
   *
   * Retrieves source information from host memory for 2D problems during
   * CPU-based processing, initialization, and analysis operations. Provides
   * access to time functions and interpolation weights for host algorithms.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source function evaluation
   * @param index Element index containing source location information
   * @param point_source [out] Output structure populated with 2D source data
   * @note This function is an implementation detail and is called by
   *       higher-level @c load_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  /**
   * @brief Load 3D source data for host-based computations
   *
   * Retrieves source information from host memory for 3D problems during
   * CPU-based processing, initialization, and analysis operations. Provides
   * access to time functions and interpolation weights for host algorithms.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source function evaluation
   * @param index Element index containing source location information
   * @param point_source [out] Output structure populated with 3D source data
   * @note This function is an implementation detail and is called by
   *       higher-level @c load_on_device function.
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  /**
   * @brief Store 2D source data from host computations
   *
   * Writes computed source data back to host memory for 2D problems.
   * Used in preprocessing, initialization, and host-based iterative
   * algorithms where source parameters are updated on the CPU.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source data storage
   * @param index Element index identifying source storage location
   * @param point_source [in] Source data to be stored in host memory
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;

  /**
   * @brief Store 3D source data from host computations
   *
   * Writes computed source data back to host memory for 3D problems.
   * Used in preprocessing, initialization, and host-based iterative
   * algorithms where source parameters are updated on the CPU.
   *
   * @tparam IndexType Point index type for element-source mapping
   * @tparam PointSourceType Point source data structure
   *
   * @param timestep Current time step index for source data storage
   * @param index Element index identifying source storage location
   * @param point_source [in] Source data to be stored in host memory
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;
  ///@}
};

/**
 * @brief Filter and sort sources by medium type
 *
 * Extracts sources belonging to a specific medium type from a heterogeneous
 * source collection. This function enables medium-specific processing by
 * grouping sources with compatible physical properties and computational
 * requirements.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 * @tparam MediumTag Target medium type (`elastic_psv`, `acoustic`, etc.)
 *
 * @param sources Complete vector of sources from all medium types
 * @param element_types Element classification data (unused but maintained for
 * API consistency)
 * @param mesh Finite element mesh information (unused but maintained for API
 * consistency)
 *
 * @return std::tuple containing:
 *         - Filtered vector of sources matching the specified medium type
 *         - Original indices of the filtered sources in the input vector
 *
 * @code
 * // Example: Extract elastic sources from mixed collection
 * auto [elastic_sources, indices] = sort_sources_per_medium<
 *     specfem::element::dimension_tag::dim2,
 *     specfem::element::medium_tag::elastic_psv>(
 *         all_sources, element_types, mesh);
 * @endcode
 *
 * @note This function is an implementation detail and should be used only
 * within @ref specfem::assembly::sources construction
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
std::tuple<
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >,
    std::vector<int> >
sort_sources_per_medium(
    const std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh) {

  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
      sorted_sources;
  std::vector<int> source_indices;

  // Loop over all sources
  for (int isource = 0; isource < sources.size(); isource++) {

    // Get the source
    const auto &source = sources[isource];

    // Get the medium tag for the source
    const specfem::element::medium_tag medium_tag = source->get_medium_tag();

    // Check if the element is in currently checked medium and add to
    // the list of sources and indices if it is.
    if (medium_tag == MediumTag) {
      sorted_sources.push_back(source);
      source_indices.push_back(isource);
    }
  }
  return std::make_tuple(sorted_sources, source_indices);
}

} // namespace specfem::assembly::sources_impl
