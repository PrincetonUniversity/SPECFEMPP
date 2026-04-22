#pragma once

#include "specfem/assembly/compute_source_array.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

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
 * @brief Helper struct for constructing multidimensional array extents
 *
 * Recursively builds pointer types for multi-rank Kokkos views.
 *
 * @tparam T Base element type
 * @tparam Rank Number of dimensions
 */
template <typename T, int Rank> struct ExtentImpl {
  using type = typename ExtentImpl<T, Rank - 1>::type *;
};

/// @brief Base case for ExtentImpl recursion
template <typename T> struct ExtentImpl<T, 0> {
  using type = T;
};

/**
 * @brief Medium-specific source data management for spectral element
 * simulations
 *
 * Manages source time functions, Lagrange interpolants, and element mappings
 * for a specific (dimension, medium) pair.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 * @tparam MediumTag Physical medium type (`elastic_psv`, `acoustic`, etc.)
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct source_medium {

  /**
   * @name Public Type Constants
   */
  ///@{

  constexpr static auto medium_tag = MediumTag;       ///< Physical medium type
  constexpr static auto dimension_tag = DimensionTag; ///< Spatial dimension
  constexpr static auto ndim =
      (DimensionTag == specfem::element::dimension_tag::dim2)
          ? 2
          : 3; ///< Number of spatial dimensions
  constexpr static int source_array_rank =
      ndim + 2; ///< Rank of source array (ndim GLL + source index + component)

  ///@}

private:
  /**
   * @name Private Type Definitions
   */
  ///@{

  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using SourceTimeFunctionView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;
  using SourceArrayView =
      Kokkos::View<typename specfem::assembly::sources_impl::ExtentImpl<
                       type_real, source_array_rank>::type,
                   Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

  constexpr static int components =
      specfem::element::attributes<dimension_tag,
                                   MediumTag>::components; ///< Number of field
                                                           ///< components for
                                                           ///< this medium

  ///@}

public:
  /**
   * @name Constructors
   */
  ///@{

  /// @brief Default constructor
  source_medium() = default;

  /**
   * @brief Construct 2D source medium from source list
   *
   * Initializes source time functions and Lagrange interpolants for all
   * sources in the specified medium. Computes source arrays at GLL points.
   *
   * @param sources Source objects for this medium
   * @param mesh Finite element mesh
   * @param jacobian_matrix Jacobian transformation matrices
   * @param element_types Element classification data
   * @param t0 Initial simulation time
   * @param dt Time step size
   * @param nsteps Total number of time steps
   */
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);

  /**
   * @brief Construct 3D source medium from source list
   *
   * Initializes source time functions and Lagrange interpolants for all
   * sources in the specified medium. Computes source arrays at GLL points.
   *
   * @param sources Source objects for this medium
   * @param mesh Finite element mesh
   * @param jacobian_matrix Jacobian transformation matrices
   * @param element_types Element classification data
   * @param t0 Initial simulation time
   * @param dt Time step size
   * @param nsteps Total number of time steps
   */
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);

  ///@}

  /**
   * @name Data Members
   */
  ///@{

  IndexView source_index_mapping; ///< Maps source index to element index
                                  ///< (device)
  IndexView::HostMirror h_source_index_mapping; ///< Maps source index to
                                                ///< element index (host)

  SourceTimeFunctionView
      source_time_function; ///< Source time function values
                            ///< [timestep][source][component]
                            ///< (device)
  SourceTimeFunctionView::HostMirror h_source_time_function; ///< Source time
                                                             ///< function
                                                             ///< values (host)

  SourceArrayView source_array; ///< Lagrange interpolant weights at GLL points
                                ///< (device)
  typename SourceArrayView::HostMirror h_source_array; ///< Lagrange interpolant
                                                       ///< weights (host)

  ///@}

  /**
   * @brief Load source data on device for 2D simulations
   *
   * Retrieves source time function and Lagrange interpolants for a specific
   * timestep and GLL point.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Target point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [out] Populated with source data
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  /**
   * @brief Load source data on device for 3D simulations
   *
   * Retrieves source time function and Lagrange interpolants for a specific
   * timestep and GLL point.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Target point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [out] Populated with source data
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  /**
   * @brief Store source data on device for 2D simulations
   *
   * Writes source time function and Lagrange interpolants to device memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Source point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [in] Source data to store
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  /**
   * @brief Store source data on device for 3D simulations
   *
   * Writes source time function and Lagrange interpolants to device memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Source point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [in] Source data to store
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  /**
   * @brief Load source data on host for 2D simulations
   *
   * Retrieves source time function and Lagrange interpolants for a specific
   * timestep and GLL point from host memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Target point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [out] Populated with source data
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  /**
   * @brief Load source data on host for 3D simulations
   *
   * Retrieves source time function and Lagrange interpolants for a specific
   * timestep and GLL point from host memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Target point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [out] Populated with source data
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  /**
   * @brief Store source data on host for 2D simulations
   *
   * Writes source time function and Lagrange interpolants to host memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Source point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [in] Source data to store
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;

  /**
   * @brief Store source data on host for 3D simulations
   *
   * Writes source time function and Lagrange interpolants to host memory.
   *
   * @tparam IndexType Index type containing source and GLL coordinates
   * @tparam PointSourceType Source point source structure
   * @param timestep Current time step index
   * @param index Source and grid point indices
   * @param point_source [in] Source data to store
   */
  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;
};

/**
 * @brief Filter and sort sources by medium type
 *
 * Extracts sources matching the specified medium tag and returns them with
 * their original indices for mapping.
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam MediumTag Target medium type to filter
 * @param sources All sources to filter
 * @param element_types Element classification data (unused but kept for
 * interface consistency)
 * @param mesh Finite element mesh (unused but kept for interface consistency)
 * @return Tuple of (filtered sources, original indices)
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

  for (int isource = 0; isource < (int)sources.size(); isource++) {
    const auto &source = sources[isource];

    // In MPI builds a source not owned by this rank has ispec = -1.
    // Skip it here so it is never added to any medium's source list.
    if (source->get_local_coordinates().ispec < 0) {
      continue;
    }
    if (source->get_medium_tag() == MediumTag) {
      sorted_sources.push_back(source);
      source_indices.push_back(isource);
    }
  }
  return std::make_tuple(sorted_sources, source_indices);
}

} // namespace specfem::assembly::sources_impl

#include "dim2/source_medium.tpp"
#include "dim3/source_medium.tpp"
