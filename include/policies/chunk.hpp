#pragma once

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

namespace specfem {
namespace iterator {

namespace impl {
template <bool UseSIMD> struct index_type;

template <> struct index_type<true> {
  int ielement;
  specfem::point::simd_index index;

  KOKKOS_INLINE_FUNCTION
  index_type(const int ielement, const specfem::point::simd_index index)
      : ielement(ielement), index(index) {}
};

template <> struct index_type<false> {
  int ielement;
  specfem::point::index index;

  KOKKOS_INLINE_FUNCTION
  index_type(const int ielement, const specfem::point::index index)
      : ielement(ielement), index(index){};
};
} // namespace impl

template <typename ViewType, typename SIMDType> class chunk {
public:
  using simd = SIMDType;
  constexpr static bool using_simd = simd::using_simd;
  constexpr static int simd_size = simd::size();
  using index_type = typename impl::index_type<simd::using_simd>;

private:
  ViewType indices;
  int num_elements;
  int ngllz;
  int ngllx;

  KOKKOS_INLINE_FUNCTION
  chunk(const ViewType &indices, const int ngllz, const int ngllx,
        std::true_type)
      : indices(indices), num_elements(indices.extent(0) / simd_size +
                                       (indices.extent(0) % simd_size != 0)),
        ngllz(ngllz), ngllx(ngllx) {}

  KOKKOS_INLINE_FUNCTION
  chunk(const ViewType &indices, const int ngllz, const int ngllx,
        std::false_type)
      : indices(indices), num_elements(indices.extent(0)), ngllz(ngllz),
        ngllx(ngllx) {}

  KOKKOS_INLINE_FUNCTION
  impl::index_type<false> operator()(const int i, std::false_type) const {
#ifdef KOKKOS_ENABLE_CUDA
    int ielement = i % num_elements;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz % ngllz;
    const int ix = xz / ngllz;
#else
    const int ix = i % ngllx;
    const int iz = (i / ngllx) % ngllz;
    const int ielement = i / (ngllz * ngllx);
    int ispec = indices(ielement);
#endif
    return impl::index_type<false>(ielement,
                                   specfem::point::index(ispec, iz, ix));
  }

  KOKKOS_INLINE_FUNCTION
  impl::index_type<true> operator()(const int i, std::true_type) const {
#ifdef KOKKOS_ENABLE_CUDA
    int ielement = i % num_elements;
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz % ngllz;
    const int ix = xz / ngllz;
#else
    const int ix = i % ngllx;
    const int iz = (i / ngllx) % ngllz;
    const int ielement = i / (ngllz * ngllx);
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
#endif
    return impl::index_type<true>(
        ielement, specfem::point::simd_index(ispec, simd_elements, iz, ix));
  }

public:
  KOKKOS_INLINE_FUNCTION
  chunk(const ViewType &indices, int ngllz, int ngllx)
      : chunk(indices, ngllz, ngllx,
              std::integral_constant<bool, using_simd>()) {
#if KOKKOS_VERSION < 40100
    static_assert(ViewType::Rank == 1, "View must be rank 1");
#else
    static_assert(ViewType::rank() == 1, "View must be rank 1");
#endif
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int chunk_size() const { return num_elements * ngllz * ngllx; }

  KOKKOS_INLINE_FUNCTION
  index_type operator()(const int i) const {
    return operator()(i, std::integral_constant<bool, using_simd>());
  }
};
} // namespace iterator

namespace policy {
/**
 * @brief Chunk policy for parallel execution
 *
 * A version of Kokkos::TeamPolicy that is used to chunk a set of elements into
 * chunks of ParallelConfig::ChunkSize. Every physical team gets a consecutive
 * chunk of elements to work on.
 *
 * For best performance, every thread in a team should work on a different
 * elements, and the data should be contiguous in element index.
 *
 * @tparam ParallelConfig Configuration of the parallel execution. Must have
 * ParallelConfig::num_threads and ParallelConfig::vector_lanes as static
 * members.
 * @tparam PolicyTraits Traits for the Kokkos::TeamPolicy
 */
template <typename ParallelConfig, typename... PolicyTraits>
struct element_chunk : public Kokkos::TeamPolicy<PolicyTraits...> {
public:
  constexpr static bool isChunkable = true; ///< Chunkable policy
  constexpr static bool isIterable = false; ///< Not iterable
  constexpr static int ChunkSize = ParallelConfig::chunk_size;   ///< Chunk size
  constexpr static int NumThreads = ParallelConfig::num_threads; ///< Chunk size
  constexpr static int VectorLanes =
      ParallelConfig::vector_lanes;                          ///< Vector lanes
  constexpr static int TileSize = ParallelConfig::tile_size; ///< Tile size

  using simd = typename ParallelConfig::simd; ///< SIMD configuration

  using PolicyType =
      Kokkos::TeamPolicy<PolicyTraits...>; ///< Kokkos::TeamPolicy type

  using member_type =
      typename PolicyType::member_type; ///< Member type of the policy

  using ViewType = Kokkos::View<
      int *, typename member_type::execution_space::memory_space>; ///< View
                                                                   ///< type
                                                                   ///< used to
                                                                   ///< store
                                                                   ///< elements

  using iterator_type =
      specfem::iterator::chunk<ViewType, simd>; ///< Iterator type

  /**
   * @brief Construct a new element chunk policy
   *
   * @param view View of elements to chunk
   */
  element_chunk(const ViewType &view, int ngllz, int ngllx)
      : PolicyType(view.extent(0) / TileSize + (view.extent(0) % TileSize != 0),
                   NumThreads, VectorLanes),
        elements(view), ngllz(ngllz), ngllx(ngllx) {
#if KOKKOS_VERSION < 40100
    static_assert(ViewType::Rank == 1, "View must be rank 1");
#else
    static_assert(ViewType::rank() == 1, "View must be rank 1");
#endif
  }

  /**
   * @brief Get the policy object
   *
   * @return PolicyType&  Reference to the policy
   */
  inline PolicyType &get_policy() { return *this; }

  /**
   * @brief Get the chunk of elements for the team
   *
   * @param team_rank Rank of the team, usually referenced by member.team_rank()
   * @return auto  Kokkos::subview of the elements for the team
   */
  KOKKOS_INLINE_FUNCTION
  auto league_iterator(const int start_index) const {
    const int start = start_index;
    const int end = (start + ChunkSize > elements.extent(0))
                        ? elements.extent(0)
                        : start + ChunkSize;
    const auto my_indices =
        Kokkos::subview(elements, Kokkos::make_pair(start, end));
    return specfem::iterator::chunk<ViewType, simd>(my_indices, ngllz, ngllx);
  }

private:
  ViewType elements; ///< View of elements
  int ngllz;
  int ngllx;
};
} // namespace policy
} // namespace specfem
