#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

// Forward declarations
namespace specfem {
namespace algorithms {
template <specfem::dimension::type DimensionTag>
specfem::point::local_coordinates<DimensionTag> locate_point(
    const specfem::point::global_coordinates<DimensionTag> &coordinates,
    const specfem::assembly::mesh<DimensionTag> &mesh);
}
} // namespace specfem

// EXTENT_IMPL_CHANGE

namespace specfem::assembly::sources_impl {
template <typename T, int Rank> struct ExtentImpl {
  using type = typename ExtentImpl<T, Rank - 1>::type *;
};

template <typename T> struct ExtentImpl<T, 0> {
  using type = T;
};

} // namespace specfem::assembly::sources_impl

namespace specfem::assembly::sources_impl {
/**
 * @brief Information about sources located within a medium
 *
 * @tparam Dimension Dimension of spectral elements
 * @tparam Medium Medium type
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct source_medium {

public:
  /**
   * @name Compile-time constants
   *
   */
  ///@{
  constexpr static auto medium_tag = MediumTag; ///< Medium type
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of spectral elements
  constexpr static auto ndim =
      (DimensionTag == specfem::dimension::type::dim2) ? 2 : 3; // 2D or 3D
  constexpr static int source_array_rank = ndim + 2; // 4 for 2D, 5 for 3D
  ///@}

private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using SourceTimeFunctionView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store source time
                                                   ///< functions
  // EXTENT_IMPL_CHANGE
  using SourceArrayView =
      Kokkos::View<typename specfem::assembly::sources_impl::ExtentImpl<
                       type_real, source_array_rank>::type,
                   Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

  constexpr static int components =
      specfem::element::attributes<dimension_tag,
                                   MediumTag>::components; ///< Number
                                                           ///< of
                                                           ///< components
                                                           ///< in the
                                                           ///< medium
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
  source_medium() = default;

  /**
   * @brief Compute source information for 2D case
   * @details Constructs source information for 2D spectral elements
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<dimension_tag> > > &sources,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);

  /**
   * @overload
   * @brief Compute source information for 3D case
   * @details Constructs source information for 3D spectral elements
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<dimension_tag> > > &sources,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const type_real t0, const type_real dt, const int nsteps);
  ///@}

  IndexView source_index_mapping; ///< Spectral element index for every source
  IndexView::HostMirror h_source_index_mapping; ///< Host mirror of
                                                ///< source_index_mapping
  SourceTimeFunctionView source_time_function;  ///< Source time function for
                                                ///< every source
  SourceTimeFunctionView::HostMirror
      h_source_time_function;   ///< Host mirror of source_time_function
  SourceArrayView source_array; ///< Lagrange interpolants for every source
  typename SourceArrayView::HostMirror h_source_array; ///< Host mirror of
                                                       ///< source_array

  /*
  // TODO(Lucas : CPP20 update)
  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim2)
  KOKKOS_INLINE_FUNCTION void load_on_device(...) const;

  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim3)
  KOKKOS_INLINE_FUNCTION void load_on_device(...) const;
  */
  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const;

  /*
  // TODO(Lucas : CPP20 update)
  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim2)
  KOKKOS_INLINE_FUNCTION void store_on_device(...) const;

  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim3)
  KOKKOS_INLINE_FUNCTION void store_on_device(...) const;
  */
  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const;

  /*
  // TODO(Lucas : CPP20 update)
  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim2)
  void load_on_host(...) const;

  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim3)
  void load_on_host(...) const;
  */
  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const;

  /*
  // TODO(Lucas : CPP20 update)
  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim2)
  */
  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;

  /*
  // TODO(Lucas : CPP20 update)
  template <typename IndexType, typename PointSourceType>
  requires (DimensionTag == specfem::dimension::type::dim3)
  */
  template <typename IndexType, typename PointSourceType,
            specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const;
};

/** @brief Sort sources per medium
 * @tparam DimensionTag Dimension tag (e.g., dim2)
 * @tparam MediumTag Medium tag (e.g., elastic_psv, acoustic, etc.)
 * @param sources Vector of sources to be sorted
 * @param element_types Element types for every spectral element
 * @param mesh Finite element mesh information
 * @return Tuple containing sorted sources and their indices
 */
template <specfem::dimension::type DimensionTag,
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
