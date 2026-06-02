#pragma once

#include "chunked_domain_iterator.hpp"
#include "policy.hpp"
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
class MappedIndex;
} // namespace chunk_element
} // namespace specfem

namespace specfem {
namespace execution {

template <specfem::element::dimension_tag DimensionTag,
          typename KokkosIndexType, bool UseSIMD, typename ExecutionSpace>
class MappedPointIndex {
private:
  using index_type =
      specfem::point::mapped_index<DimensionTag,
                                   UseSIMD>; ///< Mapped index type
  using point_index_type = specfem::point::index<DimensionTag, UseSIMD>;

public:
  using iterator_type =
      VoidIterator<ExecutionSpace>; ///< Iterator type for this index

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type get_index() const {
    return this->index; ///< Returns the point index
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const point_index_type get_local_index() const {
    return this->local_index; ///< Returns the point index
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType get_policy_index() const {
    return this->kokkos_index; ///< Returns the Kokkos index
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const {
    return iterator_type{}; ///< Returns an empty iterator
  }

  KOKKOS_INLINE_FUNCTION
  MappedPointIndex(const point_index_type &index, const int &imap,
                   const point_index_type &local_index,
                   const KokkosIndexType &kokkos_index)
      : index(index, imap), local_index(local_index),
        kokkos_index(kokkos_index) {}

  KOKKOS_INLINE_FUNCTION
  MappedPointIndex() = default;

  /// @brief Construct an "end" sentinel index — tail-skip marker for the
  /// chunked iterator.
  KOKKOS_INLINE_FUNCTION
  static MappedPointIndex end() {
    MappedPointIndex index;
    index.is_end_flag = true;
    return index;
  }

  KOKKOS_INLINE_FUNCTION
  bool is_end() const { return is_end_flag; }

private:
  index_type index;             ///< Point
  point_index_type local_index; ///< Local point index
  KokkosIndexType kokkos_index; ///< Kokkos index type
  bool is_end_flag = false;     ///< True if this is the tail-skip sentinel.
};

template <specfem::element::dimension_tag DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType,
          typename G = specfem::mesh_entity::Grid<>, int ChunkSize = 1>
class MappedChunkElementIterator
    : public ChunkElementIterator<DimensionTag, SIMD, ViewType, TeamMemberType,
                                  G, ChunkSize> {
private:
  using base_type = ChunkElementIterator<DimensionTag, SIMD, ViewType,
                                         TeamMemberType, G, ChunkSize>;
  constexpr static auto simd_size = SIMD::size();
  constexpr static auto using_simd = SIMD::using_simd;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using execution_space =
      typename base_type::execution_space; ///< Execution space type
  using index_type = MappedPointIndex<DimensionTag, policy_index_type,
                                      using_simd, execution_space>;

  KOKKOS_INLINE_FUNCTION const index_type
  operator()(const policy_index_type &i) const {
    const auto base_index = base_type::operator()(i);
    if (base_index.is_end())
      return index_type::end();
    // Calculate the mapped index
    const auto index = base_index.get_index();
    const auto local_index = base_index.get_local_index();
    return index_type(index, mapping(local_index.ispec), local_index, i);
  }

  KOKKOS_INLINE_FUNCTION
  MappedChunkElementIterator(
      const TeamMemberType &team, const ViewType indices,
      const ViewType mapping,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid)
      : base_type(team, indices, element_grid), mapping(mapping) {}

private:
  ViewType mapping;
};

template <specfem::element::dimension_tag DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType,
          typename G = specfem::mesh_entity::Grid<>, int ChunkSize = 1>
class MappedChunkElementIndex
    : public ChunkElementIndex<DimensionTag, SIMD, ViewType, TeamMemberType, G,
                               ChunkSize> {
private:
  using base_type = ChunkElementIndex<DimensionTag, SIMD, ViewType,
                                      TeamMemberType, G, ChunkSize>;
  using index_type =
      specfem::chunk_element::MappedIndex<DimensionTag, SIMD, ViewType,
                                          TeamMemberType, G,
                                          ChunkSize>; ///< Underlying index
                                                      ///< type used to
                                                      ///< store the
                                                      ///< indices within
                                                      ///< the chunk.

public:
  using iterator_type =
      MappedChunkElementIterator<DimensionTag, SIMD, ViewType, TeamMemberType,
                                 G, ChunkSize>;

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type get_index() const { return { *this }; }

  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return this->iterator; }

  KOKKOS_INLINE_FUNCTION
  MappedChunkElementIndex(
      const ViewType indices, const ViewType mapping,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid,
      const TeamMemberType &kokkos_index)
      : base_type(indices, element_grid, kokkos_index), mapping(mapping),
        iterator(kokkos_index, indices, mapping, element_grid) {}

private:
  ViewType mapping;
  iterator_type iterator;
};

template <specfem::element::dimension_tag DimensionTag, typename ParallelConfig,
          typename ViewType, typename G = specfem::mesh_entity::Grid<>>
class MappedChunkedDomainIterator
    : public ChunkedDomainIterator<DimensionTag, ParallelConfig, ViewType, G> {
private:
  using base_type =
      ChunkedDomainIterator<DimensionTag, ParallelConfig, ViewType, G>;
  constexpr static auto simd_size = ParallelConfig::simd::size();
  constexpr static auto chunk_size = ParallelConfig::chunk_size;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using index_type = MappedChunkElementIndex<
      ParallelConfig::dimension, typename ParallelConfig::simd,
      decltype(Kokkos::subview(std::declval<ViewType>(),
                               std::declval<Kokkos::pair<int, int>>())),
      policy_index_type, G, ParallelConfig::chunk_size>;
  using execution_space =
      typename base_type::execution_space; ///< Execution space type

  using base_index_type =
      MappedPointIndex<DimensionTag, typename ViewType::value_type,
                       ParallelConfig::simd::using_simd, execution_space>;

  MappedChunkedDomainIterator(
      const ViewType indices, const ViewType mapping,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid)
      : base_type(indices, element_grid), mapping(mapping) {}

  MappedChunkedDomainIterator(
      const ParallelConfig, const ViewType indices, const ViewType mapping,
      const specfem::mesh_entity::element_grid<DimensionTag, G> &element_grid)
      : MappedChunkedDomainIterator(indices, mapping, element_grid) {}

  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const int league_id = team.league_rank();
    const int start = league_id * chunk_size * simd_size;
    const int end =
        ((start + chunk_size * simd_size) > base_type::indices.extent(0))
            ? base_type::indices.extent(0)
            : start + chunk_size * simd_size;
    const auto my_indices =
        Kokkos::subview(base_type::indices, Kokkos::make_pair(start, end));
    const auto my_mapping =
        Kokkos::subview(mapping, Kokkos::make_pair(start, end));
    return index_type(my_indices, my_mapping, base_type::element_grid, team);
  }

  template <typename... Args>
  inline MappedChunkedDomainIterator &set_scratch_size(Args &&...args) {
    base_policy_type::set_scratch_size(std::forward<Args>(args)...);
    return *this; ///< Returns itself for method chaining
  }

private:
  ViewType mapping;
};

// Template argument deduction guides
template <typename ParallelConfig, typename ViewType,
          specfem::element::dimension_tag DimensionTag, typename G>
MappedChunkedDomainIterator(
    ParallelConfig, ViewType, ViewType,
    const specfem::mesh_entity::element_grid<DimensionTag, G> &)
    -> MappedChunkedDomainIterator<DimensionTag, ParallelConfig, ViewType, G>;

template <typename ViewType, specfem::element::dimension_tag DimensionTag,
          typename G>
MappedChunkedDomainIterator(
    ViewType, ViewType,
    const specfem::mesh_entity::element_grid<DimensionTag, G> &)
    -> MappedChunkedDomainIterator<
        DimensionTag,
        decltype(std::declval<typename ViewType::execution_space>()), ViewType,
        G>;

} // namespace execution
} // namespace specfem

#include "specfem/chunk_element/mapped_index.hpp"
