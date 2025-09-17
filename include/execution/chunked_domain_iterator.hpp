#pragma once

#include "macros.hpp"
#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace chunk_element {

// Forward declaration for PointIndex
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
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
template <specfem::dimension::type DimensionTag, typename KokkosIndexType,
          bool UseSIMD, typename ExecutionSpace>
class PointIndex;

/**
 * @brief 2D specialization of PointIndex
 */
template <typename KokkosIndexType, bool UseSIMD, typename ExecutionSpace>
class PointIndex<specfem::dimension::type::dim2, KokkosIndexType, UseSIMD,
                 ExecutionSpace> {
private:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
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
   * @brief Constructor for PointIndex when SIMD is used.
   *
   * @param ispec The index of the spectral element.
   * @param number_elements The number of elements in the chunk.
   * @param iz The z-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param kokkos_index The Kokkos index type.
   */
  template <bool U = UseSIMD, typename std::enable_if<U, int>::type = 0>
  KOKKOS_INLINE_FUNCTION PointIndex(const int &ispec,
                                    const int &number_elements, const int &iz,
                                    const int &ix, const int &ielement,
                                    const KokkosIndexType &kokkos_index)
      : index(ispec, number_elements, iz, ix),
        local_index(ielement, number_elements, iz, ix),
        kokkos_index(kokkos_index) {}

  /**
   * @brief Constructor for PointIndex when SIMD is not used.
   *
   * @param ispec The index of the spectral element.
   * @param iz The z-coordinate of the GLL point.
   * @param ix The x-coordinate of the GLL point.
   * @param kokkos_index The Kokkos index type.
   */
  template <bool U = UseSIMD, typename std::enable_if<!U, int>::type = 0>
  KOKKOS_INLINE_FUNCTION PointIndex(const int &ispec, const int &iz,
                                    const int &ix, const int &ielement,
                                    const KokkosIndexType &kokkos_index)
      : index(ispec, iz, ix), local_index(ielement, iz, ix),
        kokkos_index(kokkos_index) {}

  KOKKOS_INLINE_FUNCTION
  constexpr bool is_end() const {
    return false; ///< Returns false as this is not an end iterator
  }

private:
  point_index_type index;       ///< Index of the GLL
                                ///< point
  point_index_type local_index; ///< Index of the
                                ///< GLL point
                                ///< relative to
                                ///< current chunk
  KokkosIndexType kokkos_index; ///< The Kokkos index
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
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class ChunkElementIterator;

/**
 * @brief 2D specialization of ChunkElementIterator
 */
template <typename SIMD, typename ViewType, typename TeamMemberType>
class ChunkElementIterator<specfem::dimension::type::dim2, SIMD, ViewType,
                           TeamMemberType>
    : public TeamThreadRangePolicy<TeamMemberType,
                                   typename ViewType::value_type> {
private:
  using base_type =
      TeamThreadRangePolicy<TeamMemberType, typename ViewType::value_type>;
  constexpr static auto simd_size = SIMD::size();
  constexpr static auto using_simd = SIMD::using_simd;
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

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
  template <bool U = using_simd>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<U, const index_type>
  operator()(const policy_index_type &i) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % num_elements;
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz / element_grid.ngllz;
    const int ix = xz % element_grid.ngllz;
#else
    const int ngll_total = element_grid.ngllz * element_grid.ngllx;
    const int ix = i % element_grid.ngllx;
    const int iz = (i / element_grid.ngllx) % element_grid.ngllz;
    const int ielement = i / ngll_total;
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
#endif
    return index_type(ispec, simd_elements, iz, ix, ielement, i);
  }

  template <bool U = using_simd>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<!U, const index_type>
  operator()(const policy_index_type &i) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int ielement = i % num_elements;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz / element_grid.ngllz;
    const int ix = xz % element_grid.ngllz;
#else
    const int ix = i % element_grid.ngllx;
    const int iz = (i / element_grid.ngllx) % element_grid.ngllz;
    const int ielement = i / (element_grid.ngllz * element_grid.ngllx);
    int ispec = indices(ielement);
#endif
    return index_type(ispec, iz, ix, ielement, i);
  }

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
      const specfem::mesh_entity::element<dimension_tag> &element_grid)
      : indices(indices), element_grid(element_grid),
        num_elements((indices.extent(0) / simd_size) +
                     ((indices.extent(0) % simd_size) != 0)),
        base_type(team, (((indices.extent(0) / simd_size) +
                          (indices.extent(0) % simd_size != 0)) *
                         element_grid.ngllz * element_grid.ngllx)) {}

private:
  ViewType indices; ///< View of indices of elements within this chunk
  int num_elements; ///< Number of elements in the chunk, adjusted for SIMD
  specfem::mesh_entity::element<dimension_tag> element_grid; ///< Element grid
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
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class ChunkElementIndex {
private:
  using index_type =
      specfem::chunk_element::Index<DimensionTag, SIMD, ViewType,
                                    TeamMemberType>; ///< Underlying index type
                                                     ///< used to store the
                                                     ///< indices within the
                                                     ///< chunk.

public:
  using iterator_type =
      ChunkElementIterator<DimensionTag, SIMD, ViewType,
                           TeamMemberType>; ///< Iterator type for iterating
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
      const specfem::mesh_entity::element<DimensionTag> &element_grid,
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
  ViewType indices;                                         ///< View of indices
  specfem::mesh_entity::element<DimensionTag> element_grid; ///< Element grid
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
 * specfem::parallel_configuration::chunk_config
 * @tparam ViewType Type of the view containing indices of elements.
 */
template <specfem::dimension::type DimensionTag, typename ParallelConfig,
          typename ViewType>
class ChunkedDomainIterator;

template <typename ParallelConfig, typename ViewType>
class ChunkedDomainIterator<specfem::dimension::type::dim2, ParallelConfig,
                            ViewType> : public TeamPolicy<ParallelConfig> {
private:
  using base_type = TeamPolicy<ParallelConfig>; ///< Base policy type
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2;                             ///< Dimension
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
      ParallelConfig::dimension, typename ParallelConfig::simd,
      decltype(Kokkos::subview(std::declval<ViewType>(),
                               std::declval<Kokkos::pair<int, int> >())),
      policy_index_type>; ///< Underlying index type. This index
                          ///< will be passed to the closure when
                          ///< calling @ref
                          ///< specfem::execution::for_each_level
  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  /**
   * @brief Construct a new Chunked Domain Iterator object
   *
   * @param indices View of indices of elements within this iterator.
   * @param element_grid specfem::mesh_entity::element which defines the number
   *                     of GLL points etc.
   */
  ChunkedDomainIterator(
      const ViewType indices,
      const specfem::mesh_entity::element<dimension_tag> &element_grid)
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
      const specfem::mesh_entity::element<dimension_tag> &element_grid)
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

protected:
  ViewType indices; ///< View of indices of elements within this iterator
  specfem::mesh_entity::element<dimension_tag> element_grid; ///< Element grid
                                                             ///< information
};

// Template argument deduction guides
template <typename ParallelConfig, typename ViewType,
          specfem::dimension::type DimensionTag>
ChunkedDomainIterator(ParallelConfig, ViewType,
                      const specfem::mesh_entity::element<DimensionTag> &)
    -> ChunkedDomainIterator<DimensionTag, ParallelConfig, ViewType>;

template <typename ViewType, specfem::dimension::type DimensionTag>
ChunkedDomainIterator(ViewType,
                      const specfem::mesh_entity::element<DimensionTag> &)
    -> ChunkedDomainIterator<
        DimensionTag,
        decltype(std::declval<typename ViewType::execution_space>()), ViewType>;

} // namespace execution
} // namespace specfem

#include "specfem/chunk_element/index.hpp"
