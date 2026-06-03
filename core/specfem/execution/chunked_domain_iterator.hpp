#pragma once

#include "policy.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace chunk_element {

// Forward declaration for PointIndex
template <specfem::element::dimension_tag DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType,
          typename G = specfem::mesh_entity::Grid<>, int ChunkSize = 1>
class Index;
} // namespace chunk_element
} // namespace specfem

namespace specfem {
namespace execution {

/**
 * @brief PointIndex class is used to represent a quadrature point index within
 * a domain.
 *
 * @tparam DimensionTag The dimension tag (e.g., dim2, dim3).
 * @tparam KokkosIndexType The type of the Kokkos index, must be convertible to
 * an integral type.
 * @tparam UseSIMD Indicates whether SIMD is used for the index.
 * @tparam ExecutionSpace The execution space type where the index will be
 * executed.
 */
template <specfem::element::dimension_tag DimensionTag,
          typename KokkosIndexType, bool UseSIMD, typename ExecutionSpace>
class PointIndex {
private:
  constexpr static auto dimension_tag = DimensionTag;
  using point_index_type = specfem::point::index<dimension_tag, UseSIMD>;

public:
  using iterator_type =
      VoidIterator<ExecutionSpace>; ///< Iterator type used to iterate over
                                    ///< GLL points within this index.
                                    ///< @c VoidIterator is used when the
                                    ///< index refers to a single GLL point.

  /**
   * @brief Get the policy index that defined this point index.
   *
   * @return const KokkosIndexType The policy index that defined this point
   * index.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType get_policy_index() const {
    return this->kokkos_index; ///< Returns the policy index
  }

  /**
   * @brief Get the underlying index used to define the GLL point.
   *
   * @return const specfem::point::index<DimensionTag, UseSIMD> The point index
   * that defines the GLL point.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const point_index_type get_index() const {
    return this->index; ///< Returns the point index
  }

  /**
   * @brief Get the underlying index used to define the GLL point relative to
   * current chunk.
   *
   * @return const specfem::point::index<DimensionTag, UseSIMD> The point index
   * that defines the GLL point relative to current chunk.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const point_index_type get_local_index() const {
    return this->local_index; ///< Returns the local point index
  }

  /**
   * @brief Get the iterator for this index.
   *
   * @return const iterator_type The iterator for this index.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return iterator_type{}; }

  /**
   * @brief Constructor for 2D PointIndex when SIMD is used.
   *
   * @param ispec The index of the spectral element.
   * @param number_elements The number of elements in the chunk.
   * @param iz The z-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param ielement The index of the element relative to the first in the
   * chunk.
   * @param kokkos_index The Kokkos index type.
   */
  template <bool U = UseSIMD, specfem::element::dimension_tag D = DimensionTag,
            typename std::enable_if<
                U && D == specfem::element::dimension_tag::dim2, int>::type = 0>
  KOKKOS_INLINE_FUNCTION PointIndex(const int &ispec,
                                    const int &number_elements, const int &iz,
                                    const int &ix, const int &ielement,
                                    const KokkosIndexType &kokkos_index)
      : index(ispec, number_elements, iz, ix),
        local_index(ielement, number_elements, iz, ix),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Constructor for 2D PointIndex when SIMD is not used.
   *
   * @param ispec The index of the spectral element.
   * @param iz The z-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param kokkos_index The Kokkos index type.
   */
  template <
      bool U = UseSIMD, specfem::element::dimension_tag D = DimensionTag,
      typename std::enable_if<!U && D == specfem::element::dimension_tag::dim2,
                              int>::type = 0>
  KOKKOS_INLINE_FUNCTION PointIndex(const int &ispec, const int &iz,
                                    const int &ix, const int &ielement,
                                    const KokkosIndexType &kokkos_index)
      : index(ispec, iz, ix), local_index(ielement, iz, ix),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Constructor for 3D PointIndex when SIMD is used.
   *
   * @param ispec The index of the spectral element.
   * @param number_elements The number of elements in the chunk.
   * @param iz The z-coordinate of the GLL point.
   * @param iy The y-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param ielement The index of the element relative to the first in the
   * chunk.
   * @param kokkos_index The Kokkos index type.
   */
  template <bool U = UseSIMD, specfem::element::dimension_tag D = DimensionTag,
            typename std::enable_if<
                U && D == specfem::element::dimension_tag::dim3, int>::type = 0>
  KOKKOS_INLINE_FUNCTION
  PointIndex(const int &ispec, const int &number_elements, const int &iz,
             const int &iy, const int &ix, const int &ielement,
             const KokkosIndexType &kokkos_index)
      : index(ispec, number_elements, iz, iy, ix),
        local_index(ielement, number_elements, iz, iy, ix),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Constructor for 3D PointIndex when SIMD is not used.
   *
   * @param ispec The index of the spectral element.
   * @param iz The z-coordinate of the GLL point.
   * @param iy The y-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param ielement The index of the element relative to the first in the
   * chunk.
   * @param kokkos_index The Kokkos index type.
   */
  template <
      bool U = UseSIMD, specfem::element::dimension_tag D = DimensionTag,
      typename std::enable_if<!U && D == specfem::element::dimension_tag::dim3,
                              int>::type = 0>
  KOKKOS_INLINE_FUNCTION
  PointIndex(const int &ispec, const int &iz, const int &iy, const int &ix,
             const int &ielement, const KokkosIndexType &kokkos_index)
      : index(ispec, iz, iy, ix), local_index(ielement, iz, iy, ix),
        kokkos_index(kokkos_index) {}

  /// @brief Construct an "end" sentinel index. Used by the chunked iterator
  /// to signal that the policy index falls in the unused tail of the last
  /// chunk and the closure must be skipped.
  KOKKOS_INLINE_FUNCTION
  static PointIndex end() {
    PointIndex index;
    index.is_end_flag = true;
    return index;
  }

  KOKKOS_INLINE_FUNCTION
  PointIndex() = default;

  KOKKOS_INLINE_FUNCTION
  bool is_end() const { return is_end_flag; }

private:
  point_index_type index;       ///< Index of the GLL
                                ///< point
  point_index_type local_index; ///< Index of the
                                ///< GLL point
                                ///< relative to
                                ///< current chunk
  KokkosIndexType kokkos_index; ///< The Kokkos index
  bool is_end_flag = false;     ///< True if this index is the tail-skip
                                ///< sentinel.
};

/**
 * @brief ChunkElementIterator class is used to iterate over all GLL points
 * within the given chunk of elements.
 *
 * @tparam DimensionTag The dimension tag (e.g., dim2, dim3).
 * @tparam SIMD The SIMD type used for vectorization.
 * @tparam ViewType The type of the view containing indices of elements.
 * @tparam TeamMemberType The type of the Kokkos team member.
 *
 */
template <specfem::element::dimension_tag DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType,
          typename G = specfem::mesh_entity::Grid<>, int ChunkSize = 1>
class ChunkElementIterator
    : public TeamThreadRangePolicy<TeamMemberType,
                                   typename ViewType::value_type> {
private:
  using base_type =
      TeamThreadRangePolicy<TeamMemberType, typename ViewType::value_type>;
  constexpr static auto simd_size = SIMD::size();
  constexpr static auto using_simd = SIMD::using_simd;
  constexpr static auto dimension_tag = DimensionTag;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using policy_index_type =
      typename base_type::policy_index_type; /// Policy index type. Must be
                                             ///< convertible to integral type.
  using index_type =
      PointIndex<dimension_tag, policy_index_type, using_simd,
                 typename base_type::
                     execution_space>; ///< Underlying index type. This index
                                       ///< will be passed to the closure when
                                       ///< calling @ref
                                       ///< specfem::execution::for_each_level

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  /**
   * @brief Operator to get the index for a given policy index.
   *
   * @param i The policy index to convert to an index.
   * @return const index_type The index corresponding to the policy index.
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &i) const {
    return dispatch(i, G{});
  }

private:
  // ---------------------------------------------------------------
  // Dynamic-grid dispatch overloads (G = Grid<>): divisors are runtime
  // members of `element_grid`. Bodies are the existing implementation,
  // selected by SFINAE on (dimension, using_simd).
  // ---------------------------------------------------------------
  template <bool U = using_simd,
            specfem::element::dimension_tag D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<U && D == specfem::element::dimension_tag::dim2,
                              const index_type>::type
      dispatch(const policy_index_type &i, specfem::mesh_entity::Grid<>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
    int xz = i / ChunkSize;
    const int iz = xz / element_grid.ngllx;
    const int ix = xz % element_grid.ngllx;
#else
    const int ngll_total = element_grid.ngllz * element_grid.ngllx;
    const int ix = i % element_grid.ngllx;
    const int iz = (i / element_grid.ngllx) % element_grid.ngllz;
    const int ielement = i / ngll_total;
    if (ielement >= num_elements)
      return index_type::end();
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
#endif
    return index_type(ispec, simd_elements, iz, ix, ielement, i);
  }

  template <bool U = using_simd,
            specfem::element::dimension_tag D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<!U && D == specfem::element::dimension_tag::dim2,
                              const index_type>::type
      dispatch(const policy_index_type &i, specfem::mesh_entity::Grid<>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
    int xz = i / ChunkSize;
    const int iz = xz / element_grid.ngllx;
    const int ix = xz % element_grid.ngllx;
#else
    const int ix = i % element_grid.ngllx;
    const int iz = (i / element_grid.ngllx) % element_grid.ngllz;
    const int ielement = i / (element_grid.ngllz * element_grid.ngllx);
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
#endif
    return index_type(ispec, iz, ix, ielement, i);
  }

  template <bool U = using_simd,
            specfem::element::dimension_tag D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<U && D == specfem::element::dimension_tag::dim3,
                              const index_type>::type
      dispatch(const policy_index_type &i, specfem::mesh_entity::Grid<>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
    int zyx = i / ChunkSize;
    const int iz = zyx / (element_grid.nglly * element_grid.ngllx);
    const int iy = (zyx / element_grid.ngllx) % element_grid.nglly;
    const int ix = zyx % element_grid.ngllx;
#else
    const int ngll_total =
        element_grid.ngllz * element_grid.nglly * element_grid.ngllx;
    const int ix = i % element_grid.ngllx;
    const int iy = (i / element_grid.ngllx) % element_grid.nglly;
    const int iz =
        (i / (element_grid.ngllx * element_grid.nglly)) % element_grid.ngllz;
    const int ielement = i / ngll_total;
    if (ielement >= num_elements)
      return index_type::end();
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
#endif
    return index_type(ispec, simd_elements, iz, iy, ix, ielement, i);
  }

  template <bool U = using_simd,
            specfem::element::dimension_tag D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<!U && D == specfem::element::dimension_tag::dim3,
                              const index_type>::type
      dispatch(const policy_index_type &i, specfem::mesh_entity::Grid<>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
    int zyx = i / ChunkSize;
    const int iz = zyx / (element_grid.nglly * element_grid.ngllx);
    const int iy = (zyx / element_grid.ngllx) % element_grid.nglly;
    const int ix = zyx % element_grid.ngllx;
#else
    const int ix = i % element_grid.ngllx;
    const int iy = (i / element_grid.ngllx) % element_grid.nglly;
    const int iz =
        (i / (element_grid.ngllx * element_grid.nglly)) % element_grid.ngllz;
    const int ielement =
        i / (element_grid.ngllz * element_grid.nglly * element_grid.ngllx);
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
#endif
    return index_type(ispec, iz, iy, ix, ielement, i);
  }

  // ---------------------------------------------------------------
  // Compile-time-grid dispatch overloads: divisors are non-type template
  // parameters Nz/Ny/Nx so the compiler is *guaranteed* to substitute the
  // literal extent into the div/mod expressions.
  // ---------------------------------------------------------------
  template <int Nz, int Nx>
  KOKKOS_INLINE_FUNCTION const index_type dispatch(
      const policy_index_type &i, specfem::mesh_entity::Grid<Nz, Nx>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
    int xz = i / ChunkSize;
    const int iz = xz / Nx;
    const int ix = xz % Nx;
#else
    const int ix = i % Nx;
    const int iz = (i / Nx) % Nz;
    const int ielement = i / (Nz * Nx);
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
#endif
    if constexpr (using_simd) {
      int simd_elements = (simd_size + ielement > indices.extent(0))
                              ? indices.extent(0) - ielement
                              : simd_size;
      return index_type(ispec, simd_elements, iz, ix, ielement, i);
    } else {
      return index_type(ispec, iz, ix, ielement, i);
    }
  }

  // Isotropic shortcut: Grid<N> forwards to the canonical multi-arg
  // overload based on dimension_tag. N stays a non-type template parameter
  // so the literal substitution propagates through the forward.
  template <int N>
  KOKKOS_INLINE_FUNCTION const index_type
  dispatch(const policy_index_type &i, specfem::mesh_entity::Grid<N>) const {
    if constexpr (dimension_tag == specfem::element::dimension_tag::dim2) {
      return dispatch(i, specfem::mesh_entity::Grid<N, N>{});
    } else {
      return dispatch(i, specfem::mesh_entity::Grid<N, N, N>{});
    }
  }

  template <int Nz, int Ny, int Nx>
  KOKKOS_INLINE_FUNCTION const index_type
  dispatch(const policy_index_type &i,
           specfem::mesh_entity::Grid<Nz, Ny, Nx>) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % ChunkSize;
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
    int zyx = i / ChunkSize;
    const int iz = zyx / (Ny * Nx);
    const int iy = (zyx / Nx) % Ny;
    const int ix = zyx % Nx;
#else
    const int ix = i % Nx;
    const int iy = (i / Nx) % Ny;
    const int iz = (i / (Nx * Ny)) % Nz;
    const int ielement = i / (Nz * Ny * Nx);
    if (ielement >= num_elements)
      return index_type::end();
    int ispec = indices(ielement);
#endif
    if constexpr (using_simd) {
      int simd_elements = (simd_size + ielement > indices.extent(0))
                              ? indices.extent(0) - ielement
                              : simd_size;
      return index_type(ispec, simd_elements, iz, iy, ix, ielement, i);
    } else {
      return index_type(ispec, iz, iy, ix, ielement, i);
    }
  }

public:
  /**
   * @brief Constructor for ChunkElementIterator.
   *
   * @param team The Kokkos team member type.
   * @param indices View of indices of elements within this chunk.
   * @param element_grid Element grid information containing ngll, ngllx, ngllz,
   * etc.
   */
  KOKKOS_INLINE_FUNCTION ChunkElementIterator(
      const TeamMemberType &team, const ViewType indices,
      const specfem::mesh_entity::element_grid<dimension_tag, G> &element_grid)
      : indices(indices), element_grid(element_grid),
        num_elements((indices.extent(0) / simd_size) +
                     ((indices.extent(0) % simd_size) != 0)),
        base_type(team, ChunkSize * element_grid.size) {}

private:
  ViewType indices; ///< View of indices of elements within this chunk
  int num_elements; ///< Number of elements in the chunk, adjusted for SIMD
  specfem::mesh_entity::element_grid<dimension_tag, G>
      element_grid; ///< Element grid
                    ///< information
};

/**
 * @brief ChunkElementIndex class is used to represent an index for a chunk of
 * elements in a finite element mesh.
 *
 * This class provides methods to access the Kokkos index, the underlying
 * chunked element index, and an iterator for iterating over the elements in
 * the chunk.
 *
 * @tparam DimensionTag The dimension tag (e.g., dim2, dim3).
 * @tparam SIMD The SIMD type used for vectorization.
 * @tparam ViewType The type of the view containing indices of elements.
 * @tparam TeamMemberType The type of the Kokkos team member.
 */
template <specfem::element::dimension_tag DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType,
          typename G = specfem::mesh_entity::Grid<>, int ChunkSize = 1>
class ChunkElementIndex {
private:
  using index_type =
      specfem::chunk_element::Index<DimensionTag, SIMD, ViewType,
                                    TeamMemberType, G,
                                    ChunkSize>; ///< Underlying
                                                ///< index
                                                ///< type for
                                                ///< the
                                                ///< chunk element index.

public:
  using iterator_type =
      ChunkElementIterator<DimensionTag, SIMD, ViewType, TeamMemberType, G,
                           ChunkSize>; ///< Iterator type for iterating
                                       ///< over the elements in the chunk.

  /**
   * @brief Get the policy index that defined this chunk element index.
   *
   * @return const TeamMemberType The policy index that defined this chunk
   * element index.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const TeamMemberType get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Get the underlying index used to store elements in the chunk.
   *
   * @return const index_type The underlying index that defines the chunk
   * elements.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const index_type get_index() const { return { *this }; }

  /**
   * @brief Get the iterator used to iterate over the elements in the chunk.
   *
   * @return const iterator_type The iterator for this chunk element index.
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return this->iterator; }

  /**
   * @brief Constructor for ChunkElementIndex.
   *
   * @param indices View of indices of elements within this chunk.
   * @param element_grid Element grid information containing ngll, ngllx, ngllz,
   * etc.
   * @param kokkos_index Kokkos index type.
   */
  KOKKOS_INLINE_FUNCTION
  ChunkElementIndex(
      const ViewType indices,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid,
      const TeamMemberType &kokkos_index)
      : indices(indices), element_grid(element_grid),
        kokkos_index(kokkos_index),
        iterator(kokkos_index, indices, element_grid) {}

  /**
   * @brief Get a pair representing the range of indices in this chunk.
   *
   * @return Kokkos::pair<typename ViewType::value_type,
   * typename ViewType::value_type> A pair containing the first and last index
   * in the chunk.
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<typename ViewType::value_type, typename ViewType::value_type>
  get_range() const {
    return Kokkos::make_pair(indices(0),
                             indices(indices.extent(0) - 1) +
                                 1); ///< Returns the range of indices
  }

private:
  ViewType indices; ///< View of indices
  specfem::mesh_entity::element_grid<DimensionTag, G>
      element_grid;            ///< Element grid
                               ///< information
  TeamMemberType kokkos_index; ///< Kokkos index type
  iterator_type iterator;      ///< Iterator for iterating over the elements
                               ///< in the chunk.
};

/**
 * @brief ChunkedDomainIterator class is used to iterate over all quadrature
 * points within a given set of finite elements.
 *
 * The iterator divides the elements into chunks of size
 * `ParallelConfig::chunk_size` and iterates over them in parallel.
 * @tparam DimensionTag The dimension tag for the elements.
 * @tparam ParallelConfig Configuration for parallel execution. @ref
 * specfem::parallel_configurationuration::chunk_config
 * @tparam ViewType Type of the view containing indices of elements.
 */
template <specfem::element::dimension_tag DimensionTag, typename ParallelConfig,
          typename ViewType, typename G = specfem::mesh_entity::Grid<>>
class ChunkedDomainIterator : public TeamPolicy<ParallelConfig> {
private:
  using base_type = TeamPolicy<ParallelConfig>;       ///< Base policy type
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension
  constexpr static auto simd_size = ParallelConfig::simd::size(); ///< SIMD size
  constexpr static auto chunk_size = ParallelConfig::chunk_size; ///< Chunk size

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamPolicy`
  using policy_index_type = typename base_type::
      policy_index_type; ///< Policy index type. Evaluates to
                         ///< @c Kokkos::TeamPolicy::member_type
  using index_type = ChunkElementIndex<
      DimensionTag, typename ParallelConfig::simd,
      decltype(Kokkos::subview(std::declval<ViewType>(),
                               std::declval<Kokkos::pair<int, int>>())),
      policy_index_type, G,
      ParallelConfig::chunk_size>; ///< Underlying index type. This index
                                   ///< will be passed to the closure when
                                   ///< calling @ref
                                   ///< specfem::execution::for_each_level
  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  using base_index_type = PointIndex<
      dimension_tag, typename ViewType::value_type,
      ParallelConfig::simd::using_simd,
      typename base_type::execution_space>; ///< Index type to be
                                            ///< used when calling
                                            ///< @ref
                                            ///< specfem::execution::for_all
                                            ///< with this iterator.

  /**
   * @brief Construct a new Chunked Domain Iterator object
   *
   * @param indices View of indices of elements within this iterator.
   * @param element_grid specfem::mesh_entity::element which defines the number
   *                     of GLL points etc.
   */
  ChunkedDomainIterator(
      const ViewType indices,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid)
      : indices(indices), element_grid(element_grid),
        base_type(((indices.extent(0) / (chunk_size * simd_size)) +
                   ((indices.extent(0) % (chunk_size * simd_size)) != 0)),
                  Kokkos::AUTO, Kokkos::AUTO) {}

  /**
   * @brief Construct a new Chunked Domain Iterator object for a given parallel
   * configuration
   *
   * @param ParallelConfig Configuration for parallel execution.
   * @param indices View of indices of elements within this iterator.
   * @param element_grid specfem::mesh_entity::element which defines the number
   *                     of GLL points etc.
   */
  ChunkedDomainIterator(
      const ParallelConfig, const ViewType indices,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid)
      : ChunkedDomainIterator(indices, element_grid) {}

  /**
   * @brief Compute the index for a given policy index.
   *
   * The operator assigns a range of elements to each team.
   *
   * @param team The policy index for which to compute the chunked domain index.
   * @return index_type The computed index type.
   */
  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const int league_id = team.league_rank();
    const int start = league_id * chunk_size * simd_size;
    const int end = ((start + chunk_size * simd_size) > indices.extent(0))
                        ? indices.extent(0)
                        : start + chunk_size * simd_size;
    const auto my_indices =
        Kokkos::subview(indices, Kokkos::make_pair(start, end));
    return index_type(my_indices, element_grid, team);
  }

  /**
   * @brief Set scratch size for the iterator.
   *
   * This method sets passes its arguments to Kokkos::TeamPolicy's
   * set_scratch_size method, allowing for the configuration of scratch space
   * for the iterator.
   *
   * @tparam Args Types of arguments to be forwarded to the base policy.
   * @param args Arguments to be forwarded to the base policy's
   * set_scratch_size method.
   * @return ChunkedDomainIterator& Returns a reference to itself.
   */
  template <typename... Args>
  inline ChunkedDomainIterator &set_scratch_size(Args &&...args) {
    base_policy_type::set_scratch_size(std::forward<Args>(args)...);
    return *this; ///< Returns itself for method chaining
  }

  /**
   * @brief Get the maximum scratch size for the given level.
   *
   * This method returns the maximum scratch size for the given level by
   * calling the base policy's scratch_size_max method.
   *
   * @param level The level for which to get the maximum scratch size.
   * @return int The maximum scratch size for the given level.
   */
  static int scratch_size_max(int level) {
    return base_policy_type::scratch_size_max(
        level); ///< Returns the maximum scratch size for the given level
  }

protected:
  ViewType indices; ///< View of indices of elements within this iterator
  specfem::mesh_entity::element_grid<DimensionTag, G>
      element_grid; ///< Element grid
                    ///< information
};

// Template argument deduction guides
template <typename ParallelConfig, typename ViewType,
          specfem::element::dimension_tag DimensionTag, typename G>
ChunkedDomainIterator(
    ParallelConfig, ViewType,
    const specfem::mesh_entity::element_grid<DimensionTag, G> &)
    -> ChunkedDomainIterator<DimensionTag, ParallelConfig, ViewType, G>;

template <typename ViewType, specfem::element::dimension_tag DimensionTag,
          typename G>
ChunkedDomainIterator(
    ViewType, const specfem::mesh_entity::element_grid<DimensionTag, G> &)
    -> ChunkedDomainIterator<
        DimensionTag,
        decltype(std::declval<typename ViewType::execution_space>()), ViewType,
        G>;

} // namespace execution
} // namespace specfem

#include "specfem/chunk_element/index.hpp"
