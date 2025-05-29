#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

namespace specfem {
namespace policy {

namespace impl {

enum class PolicyType { KokkosPolicy, TilePolicy, VoidPolicy };

template <typename ParallelConfig>
class RangePolicy
    : public Kokkos::RangePolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type =
      Kokkos::RangePolicy<typename ParallelConfig::execution_space>;
  using policy_index_type = typename base_policy_type::index_type;

  using base_policy_type::base_policy_type;
};

template <typename ParallelConfig>
class TeamPolicy
    : public Kokkos::TeamPolicy<typename ParallelConfig::execution_space> {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type =
      Kokkos::TeamPolicy<typename ParallelConfig::execution_space>;
  using policy_index_type = typename base_policy_type::member_type;

  using base_policy_type::base_policy_type;
};

template <typename TeamMemberType, typename IndexType>
class TeamThreadRangePolicy
    : public decltype(Kokkos::TeamThreadRange(std::declval<TeamMemberType>(),
                                              std::declval<IndexType>())) {
public:
  constexpr static PolicyType policy_type = PolicyType::KokkosPolicy;
  using base_policy_type = decltype(Kokkos::TeamThreadRange(
      std::declval<TeamMemberType>(), std::declval<IndexType>()));
  using policy_index_type = IndexType;

  TeamThreadRangePolicy(const TeamMemberType &team, const IndexType &range)
      : base_policy_type(Kokkos::TeamThreadRange(team, range)) {}
};

template <std::size_t TileSize> class TeamTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = TeamTilePolicy<TileSize>;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;
};

template <std::size_t TileSize> class RangeTilePolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::TilePolicy;
  using base_policy_type = RangeTilePolicy<TileSize>;
  using policy_index_type = std::size_t;
  constexpr static std::size_t tile_size = TileSize;
};

class VoidPolicy {
public:
  constexpr static PolicyType policy_type = PolicyType::VoidPolicy;
  using base_policyType = VoidPolicy;
  using policy_index_type = void; ///< No index type for void policy
};

class VoidIterator : public VoidPolicy {
public:
  using base_policy_type = VoidPolicy;
  using policy_index_type = void;
  using index_type = void;

  KOKKOS_INLINE_FUNCTION
  constexpr index_type operator()() const { return; }
};

} // namespace impl
} // namespace policy

namespace execution {

template <typename Iterator, typename ClosureType>
inline void for_each(const std::string &name, const Iterator &iterator,
                     const ClosureType &closure) {

  static_assert(Iterator::policy_type ==
                    specfem::policy::impl::PolicyType::KokkosPolicy,
                "Iterator must be a Kokkos policy type");

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

  Kokkos::parallel_for(
      name, static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<Iterator::policy_type ==
                         specfem::policy::impl::PolicyType::KokkosPolicy,
                     void>
    for_each(const Iterator &iterator, const ClosureType &closure) {

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

  Kokkos::parallel_for(
      static_cast<const typename Iterator::base_policy_type &>(iterator),
      KOKKOS_LAMBDA(const typename Iterator::policy_index_type &i) {
        const typename Iterator::index_type index = iterator(i);
        closure(index);
      });
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<Iterator::policy_type ==
                         specfem::policy::impl::PolicyType::TilePolicy,
                     void>
    for_each(const Iterator &iterator, const ClosureType &closure) {

  constexpr std::size_t tile_size = Iterator::tile_size;

  static_assert(
      std::is_invocable_v<ClosureType, typename Iterator::index_type>,
      "Closure must be invocable with the kokkos_index_type of the iterator");

  // check if the iterator has function call operator
  static_assert(
      std::is_same_v<decltype(iterator(
                         std::declval<typename Iterator::policy_index_type>())),
                     typename Iterator::index_type>,
      "Iterator must have a function call operator that returns the "
      "index_type");

#pragma unroll
  for (std::size_t itile = 0; itile < tile_size; ++itile) {
    const typename Iterator::index_type index = iterator(itile);

    // Check if the index is valid
    if (index.is_end()) {
      break; // Skip if the index is at the end
    }

    closure(index);
  }
}

template <typename Iterator, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<Iterator::policy_type ==
                         specfem::policy::impl::PolicyType::VoidPolicy,
                     void>
    for_each(const Iterator &iterator, const ClosureType &closure) {
  static_assert(Iterator::policy_type ==
                    specfem::policy::impl::PolicyType::VoidPolicy,
                "Calling for_each on a VoidPolicy is not allowed");
}

template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<IndexType::iterator_type::policy_type ==
                         specfem::policy::impl::PolicyType::VoidPolicy,
                     void>
    for_all(const IndexType &index, const ClosureType &closure) {
  const auto i = index.get_index();
  closure(i);
}

template <typename IndexType, typename ClosureType>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<IndexType::iterator_type::policy_type !=
                         specfem::policy::impl::PolicyType::VoidPolicy,
                     void>
    for_all(const IndexType &index, const ClosureType &closure) {

  const auto iterator = index.get_iterator();

  for_each(iterator,
           [=](const typename decltype(iterator)::index_type &iter_index) {
             for_all(iter_index, closure);
           });
}

template <typename Iterator, typename ClosureType>
inline std::enable_if_t<Iterator::policy_type ==
                            specfem::policy::impl::PolicyType::KokkosPolicy,
                        void>
for_all(const std::string &name, const Iterator &iterator,
        const ClosureType &closure) {

  for_each(
      name, iterator,
      KOKKOS_LAMBDA(const typename Iterator::index_type &iter_index) {
        for_all(iter_index, closure);
      });
}
} // namespace execution

} // namespace specfem

// template <specfem::dimension::type DimensionTag, typename KokkosIndexType>
// class PointIndex {
// public:
//   const KokkosIndexType get_kokkos_index() const { return this->kokkos_index;
//   }

//   const specfem::point::index<specfem::dimension::type::dim2>
//   get_index() const {
//     return this->index;
//   }

//   constexpr auto get_iterator() const {
//     return VoidIterator(); ///< Returns a void iterator
//   }

// private:
//   specfem::point::index<specfem::dimension::type::dim2> index; ///< Point
//                                                                ///< index
//   KokkosIndexType kokkos_index; ///< Kokkos index type
// };

// template <typename KokkosIndexType> class ChunkElementIndex {
// public:
//   const KokkosIndexType get_kokkos_index() const { return this->kokkos_index;
//   }

//   const auto get_iterator() const {
//     return ChunkElementIterator(this->kokkos_index, this->chunk_size);
//   }

//   const ChunkElementIndex<KokkosIndexType> get_index() const {
//     return *this; ///< Returns itself as the index
//   }

// private:
//   KokkosIndexType kokkos_index;
//   std::size_t chunk_size;
// };

// template <typename KokkosIndexType> struct ChunkElementIndex {
//   kokkos_index_type kokkos_index; ///< Kokkos index type

//   using kokkos_index_type = KokkosIndexType; ///< Kokkos index

//   kokkos_index_type get_kokkos_index() const { return this->kokkos_index; }

//   const auto get_iterator() const {
//     return ChunkElementIterator(this->kokkos_index, this->chunk_size);
//   }

// private:
//   std::size_t chunk_size;
// }

// template <typename ParallelConfig>
// class ChunkDomainIterator : public TeamPolicy<ParallelConfig> {
// private:
//   using base_type = TeamPolicy<ParallelConfig>;

// public:
//   using index_type = ChunkElementTileIterator;
//   using base_policy_type = typename base_type::base_policy_type;
//   using policy_index_type = typename base_type::policy_index_type;

//   KOKKOS_KOKKOS_FORCEINLINE_FUNCTION_FUNCTION
//   index_type operator()(const kokkos_index_type &i) const {}
// };

// template <typename TeamMemberType, typename IndexType>
// class ChunkElementTileIterator : public TeamTilePolicy {
// private:
//   using base_type = TeamTilePolicy;

// public:
//   using base_policy_type = typename base_type::base_policy_type;
//   using policy_index_type = typename base_type::policy_index_type;
//   using index_type = ChunkElementIndex<TeamMemberType, IndexType>;

//   KOKKOS_KOKKOS_FORCEINLINE_FUNCTION_FUNCTION
//   index_type operator()(const policy_index_type &i) const {}
// };

// template <typename TeamMemberType, typename IndexType>
// class ChunkElementIterator
//     : public TeamThreadRangePolicy<TeamMemberType, IndexType> {
// private:
//   using base_type = TeamThreadRangePolicy<TeamMemberType, IndexType>;

// public:
//   using base_policy_type = typename base_type::base_policy_type;
//   using policy_index_type = typename base_type::policy_index_type;
//   using index_type = PointIndex<specfem::dimension::type::dim2>;

//   KOKKOS_KOKKOS_FORCEINLINE_FUNCTION_FUNCTION
//   index_type operator()(const policy_index_type &i) const {}
// };
