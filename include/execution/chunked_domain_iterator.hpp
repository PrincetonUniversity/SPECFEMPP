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

template <specfem::dimension::type DimensionTag, typename KokkosIndexType,
          bool UseSIMD>
class PointIndex {
public:
  using iterator_type = VoidIterator;

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType get_policy_index() const {
    return this->kokkos_index;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const specfem::point::index<DimensionTag, UseSIMD>
  get_index() const {
    return this->index; ///< Returns the point index
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const {
    return VoidIterator{}; ///< Returns a VoidIterator
  }

  template <bool U = UseSIMD, typename std::enable_if<U, int>::type = 0>
  KOKKOS_INLINE_FUNCTION
  PointIndex(const int &ispec, const int &number_elements, const int &iz,
             const int &ix, const KokkosIndexType &kokkos_index)
      : index(ispec, number_elements, iz, ix), kokkos_index(kokkos_index) {}

  template <bool U = UseSIMD, typename std::enable_if<!U, int>::type = 0>
  KOKKOS_INLINE_FUNCTION PointIndex(const int &ispec, const int &iz,
                                    const int &ix,
                                    const KokkosIndexType &kokkos_index)
      : index(ispec, iz, ix), kokkos_index(kokkos_index) {}

  KOKKOS_INLINE_FUNCTION
  constexpr bool is_end() const {
    return false; ///< Returns false as this is not an end iterator
  }

private:
  specfem::point::index<DimensionTag, UseSIMD> index; ///< Point
  KokkosIndexType kokkos_index;                       ///< Kokkos index type
};

template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class ChunkElementIterator
    : public TeamThreadRangePolicy<TeamMemberType,
                                   typename ViewType::value_type> {
private:
  using base_type =
      TeamThreadRangePolicy<TeamMemberType, typename ViewType::value_type>;
  constexpr static auto simd_size = SIMD::size();
  constexpr static auto using_simd = SIMD::using_simd;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using index_type = PointIndex<DimensionTag, policy_index_type, using_simd>;

  template <bool U = using_simd>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<U, const index_type>
  operator()(const policy_index_type &i) const {
#ifdef KOKKOS_ENABLE_CUDA
    int ielement = i % num_elements;
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz / ngllz;
    const int ix = xz % ngllz;
#else
    const int ngll_total = ngllz * ngllx;
    const int ix = i % ngllx;
    const int iz = (i / ngllx) % ngllz;
    const int ielement = i / ngll_total;
    int simd_elements = (simd_size + ielement > indices.extent(0))
                            ? indices.extent(0) - ielement
                            : simd_size;
    int ispec = indices(ielement);
#endif
    return index_type(ispec, simd_elements, iz, ix, ielement);
  }

  template <bool U = using_simd>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<!U, const index_type>
  operator()(const policy_index_type &i) const {
#ifdef KOKKOS_ENABLE_CUDA
    int ielement = i % num_elements;
    int ispec = indices(ielement);
    int xz = i / num_elements;
    const int iz = xz / ngllz;
    const int ix = xz % ngllz;
#else
    const int ix = i % ngllx;
    const int iz = (i / ngllx) % ngllz;
    const int ielement = i / (ngllz * ngllx);
    int ispec = indices(ielement);
#endif
    return index_type(ispec, iz, ix, ielement);
  }

  KOKKOS_INLINE_FUNCTION ChunkElementIterator(const TeamMemberType &team,
                                              const ViewType indices, int ngllz,
                                              int ngllx)
      : indices(indices), ngllz(ngllz), ngllx(ngllx),
        num_elements((indices.extent(0) / simd_size) +
                     ((indices.extent(0) % simd_size) != 0)),
        base_type(team, (((indices.extent(0) / simd_size) +
                          (indices.extent(0) % simd_size != 0)) *
                         ngllz * ngllx)) {}

private:
  ViewType indices;
  int num_elements;
  int ngllz;
  int ngllx;
};

template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class ChunkElementIndex {
private:
  using index_type = specfem::chunk_element::Index<DimensionTag, SIMD, ViewType,
                                                   TeamMemberType>;

public:
  using iterator_type =
      ChunkElementIterator<DimensionTag, SIMD, ViewType, TeamMemberType>;

  KOKKOS_INLINE_FUNCTION
  constexpr const TeamMemberType get_policy_index() const {
    return this->kokkos_index; ///< Returns the Kokkos index
  }

  KOKKOS_INLINE_FUNCTION
  constexpr const index_type get_index() const { return { *this }; }

  KOKKOS_INLINE_FUNCTION
  constexpr const iterator_type get_iterator() const { return this->iterator; }

  KOKKOS_INLINE_FUNCTION
  ChunkElementIndex(const ViewType indices, const int &ngllz, const int &ngllx,
                    const TeamMemberType &kokkos_index)
      : indices(indices), ngllz(ngllz), ngllx(ngllx),
        kokkos_index(kokkos_index),
        iterator(kokkos_index, indices, ngllz, ngllx) {}

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<typename ViewType::value_type, typename ViewType::value_type>
  get_range() const {
    return Kokkos::make_pair(indices(0),
                             indices(indices.extent(0) - 1) +
                                 1); ///< Returns the range of indices
  }

private:
  ViewType indices;            ///< View of indices
  int ngllz;                   ///< Number of GLL points in the z-direction
  int ngllx;                   ///< Number of GLL points in the x-direction
  TeamMemberType kokkos_index; ///< Kokkos index type
  iterator_type iterator;
};

template <typename ParallelConfig, typename ViewType>
class ChunkedDomainIterator : public TeamPolicy<ParallelConfig> {
private:
  using base_type = TeamPolicy<ParallelConfig>;
  constexpr static auto simd_size = ParallelConfig::simd::size();
  constexpr static auto chunk_size = ParallelConfig::chunk_size;

public:
  using base_policy_type = typename base_type::base_policy_type;
  using policy_index_type = typename base_type::policy_index_type;
  using index_type = ChunkElementIndex<
      ParallelConfig::dimension, typename ParallelConfig::simd,
      decltype(Kokkos::subview(std::declval<ViewType>(),
                               std::declval<Kokkos::pair<int, int> >())),
      policy_index_type>;

  ChunkedDomainIterator(const ViewType indices, int ngllz, int ngllx)
      : indices(indices), ngllz(ngllz), ngllx(ngllx),
        base_type(((indices.extent(0) / (chunk_size * simd_size)) +
                   ((indices.extent(0) % (chunk_size * simd_size)) != 0)),
                  Kokkos::AUTO, Kokkos::AUTO) {}

  ChunkedDomainIterator(const ParallelConfig, const ViewType indices, int ngllz,
                        int ngllx)
      : ChunkedDomainIterator(indices, ngllz, ngllx) {}

  KOKKOS_INLINE_FUNCTION
  const index_type operator()(const policy_index_type &team) const {
    const int league_id = team.league_rank();
    const int start = league_id * chunk_size * simd_size;
    const int end = ((start + chunk_size * simd_size) > indices.extent(0))
                        ? indices.extent(0)
                        : start + chunk_size * simd_size;
    const auto my_indices =
        Kokkos::subview(indices, Kokkos::make_pair(start, end));
    return index_type(my_indices, ngllz, ngllx, team);
  }

  template <typename... Args>
  inline ChunkedDomainIterator &set_scratch_size(Args &&...args) {
    base_policy_type::set_scratch_size(std::forward<Args>(args)...);
    return *this; ///< Returns itself for method chaining
  }

protected:
  ViewType indices; ///< View of indices of elements within this iterator
  int ngllz;        ///< Number of GLL points in the z-direction
  int ngllx;        ///< Number of GLL points in the x-direction
};

} // namespace execution
} // namespace specfem

#include "chunk_element/index.hpp"
