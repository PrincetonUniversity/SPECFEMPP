#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

namespace specfem {
namespace execution {

// ---------------------------------------------------------------------------
// Type trait: does InnerIterator expose a side-effect-free base_lookup(i)?
// PrefetchAheadIterator always defines it; plain ChunkElementIterator does not.
// ---------------------------------------------------------------------------

/**
 * @brief Type trait: check if iterator provides base_lookup() method.
 *
 * Used to distinguish prefetch-decorated iterators from plain iterators.
 * Enables safe composition of multiple prefetch decorators.
 *
 * @tparam T Iterator type
 */
template <typename T, typename = void>
struct has_base_lookup : std::false_type {};

/**
 * @brief Specialization: true if T has base_lookup() method.
 */
template <typename T>
struct has_base_lookup<
    T, std::void_t<decltype(std::declval<const T &>().base_lookup(
           std::declval<typename T::policy_index_type>()))>> : std::true_type {
};

#ifndef KOKKOS_ENABLE_CUDA

// ---------------------------------------------------------------------------
// PrefetchAheadIterator<K, InnerIterator, PrefetchFn>
//
// Inherits InnerIterator — all type aliases, policy_type, tile_size, the
// implicit cast to base_policy_type, and constructors come for free.
// Overrides operator()(i) to fire pfetch_fn(base(i+K)) as a side effect,
// then returns InnerIterator::operator()(i) unchanged.
//
// base_lookup(i) provides a side-effect-free path to the bottommost
// ChunkElementIterator — used by outer wrappers so that their lookahead
// target does not accidentally trigger inner decoration chains.
// ---------------------------------------------------------------------------

/**
 * @brief Iterator decorator that issues prefetch requests ahead of iteration.
 *
 * Wraps an inner iterator and fires a prefetch function for index i+Lookahead
 * while computing results for index i. Enables cache-efficient data access
 * patterns. Transparent pass-through on GPU.
 *
 * @tparam Lookahead Number of iterations to prefetch ahead
 * @tparam InnerIterator Base iterator type to decorate
 * @tparam PrefetchFn Callable that takes an index and issues prefetch
 *
 * @code
 * auto iter = PrefetchAheadIterator<1, BaseIter, PrefetchFn>(base, pfn);
 * for (int i = 0; i < N; ++i)
 *   process(iter(i)); // prefetches i+1 as side effect
 * @endcode
 */
template <int Lookahead, typename InnerIterator, typename PrefetchFn>
class PrefetchAheadIterator : public InnerIterator {
public:
  using index_type = typename InnerIterator::index_type; // unchanged

  KOKKOS_FORCEINLINE_FUNCTION
  const index_type
  base_lookup(const typename InnerIterator::policy_index_type &i) const {
    if constexpr (has_base_lookup<InnerIterator>::value)
      return InnerIterator::base_lookup(i);
    else
      return InnerIterator::operator()(i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const index_type
  operator()(const typename InnerIterator::policy_index_type &i) const {
    pfetch_fn(base_lookup(i + Lookahead));
    return InnerIterator::operator()(i);
  }

  PrefetchAheadIterator(const InnerIterator &inner, PrefetchFn pfetch_fn)
      : InnerIterator(inner), pfetch_fn(std::move(pfetch_fn)) {}

private:
  PrefetchFn pfetch_fn;
};

// ---------------------------------------------------------------------------
// PrefetchChunkIndex<K, InnerChunkIndex, PrefetchFn>
//
// Wraps a ChunkElementIndex (or another PrefetchChunkIndex for nesting).
// get_iterator() returns a PrefetchAheadIterator built from the inner
// iterator and the stored prefetch function.
// ---------------------------------------------------------------------------

/**
 * @brief Chunk index wrapper that decorates iterators with prefetch hints.
 *
 * Composes with ChunkElementIndex to add prefetch decoration to the returned
 * iterator. Allows stacking multiple prefetch decorators for different data
 * or lookahead distances.
 *
 * @tparam Lookahead Number of iterations to prefetch ahead
 * @tparam InnerChunkIndex Base chunk index to decorate
 * @tparam PrefetchFn Callable that prefetches data for a given index
 *
 * @code
 * auto chunk = PrefetchChunkIndex<1, BaseChunk, PrefetchFn>(base, fn);
 * // chunk.get_iterator() returns decorated iterator with prefetch
 * @endcode
 */
template <int Lookahead, typename InnerChunkIndex, typename PrefetchFn>
class PrefetchChunkIndex {
public:
  using iterator_type =
      PrefetchAheadIterator<Lookahead, typename InnerChunkIndex::iterator_type,
                            PrefetchFn>;

  KOKKOS_INLINE_FUNCTION
  iterator_type get_iterator() const {
    return { inner.get_iterator(), pfetch_fn };
  }

  KOKKOS_INLINE_FUNCTION
  constexpr auto get_policy_index() const { return inner.get_policy_index(); }

  KOKKOS_INLINE_FUNCTION
  constexpr auto get_index() const { return inner.get_index(); }

  KOKKOS_INLINE_FUNCTION
  auto get_range() const { return inner.get_range(); }

  PrefetchChunkIndex(const InnerChunkIndex &inner, PrefetchFn pfetch_fn)
      : inner(inner), pfetch_fn(std::move(pfetch_fn)) {}

private:
  InnerChunkIndex inner;
  PrefetchFn pfetch_fn;
};

/**
 * @brief Decorate a chunk index with lookahead prefetch hints.
 *
 * Returns a PrefetchChunkIndex whose iterator fires
 * prefetch_fn(base(i+Lookahead)) while computing results for index i. The
 * base() accessor bypasses nested decorations, enabling safe composition.
 *
 * Multiple decorators can be stacked for different data:
 * @code
 * prefetch_ahead<1>(prefetch_ahead<1>(chunk, F_jac), F_props)
 * // prefetches F_jac(i+1) and F_props(i+1) while computing i
 * @endcode
 *
 * No-op on GPU (KOKKOS_ENABLE_CUDA).
 *
 * @tparam Lookahead Iterations to prefetch ahead (default 1)
 * @tparam ChunkIndexType Chunk index type to decorate
 * @tparam PrefetchFn Callable(index) that issues prefetch
 * @param chunk Chunk index to decorate
 * @param fn Prefetch function
 * @return Decorated chunk index with prefetch behavior
 *
 * @ingroup Execution
 */
template <int Lookahead = 1, typename ChunkIndexType, typename PrefetchFn>
auto prefetch_ahead(const ChunkIndexType &chunk, PrefetchFn &&fn) {
  return PrefetchChunkIndex<Lookahead, ChunkIndexType,
                            std::decay_t<PrefetchFn>>{
    chunk, std::forward<PrefetchFn>(fn)
  };
}

#else // KOKKOS_ENABLE_CUDA

// GPU: __builtin_prefetch is a no-op; return a plain copy so the KOKKOS_LAMBDA
// in for_each_level captures only device-safe types.
template <int Lookahead = 1, typename ChunkIndexType, typename PrefetchFn>
ChunkIndexType prefetch_ahead(const ChunkIndexType &chunk, const PrefetchFn &) {
  return chunk;
}

#endif // KOKKOS_ENABLE_CUDA

} // namespace execution
} // namespace specfem
