#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename T>
concept Index = std::integral<T> && !std::same_as<T, bool>;

template <typename T>
concept Constant =
    Index<typename T::value_type> &&
    std::same_as<T, std::integral_constant<typename T::value_type, T::value>>;

template <typename... Values> struct Tuple;

// Storage for element I of the tuple Owner. The owner tag gives every leaf a
// distinct type, so empty leaves never share a type and never force padding.
template <typename Owner, std::size_t I, typename T,
          bool Empty = std::is_empty_v<T>>
struct TupleLeaf {
  T value{};
};

// Empty elements (integral constants, fully static subtuples) carry no state.
// They store nothing and are rebuilt from their type on access.
template <typename Owner, std::size_t I, typename T>
struct TupleLeaf<Owner, I, T, true> {
  KOKKOS_DEFAULTED_FUNCTION constexpr TupleLeaf() = default;
  KOKKOS_INLINE_FUNCTION constexpr explicit TupleLeaf(T) {}
};

template <typename Indices, typename... Values> struct TupleStorage;

// A separate empty specialization keeps the element constructor below from
// declaring a second default constructor, which nvcc rejects as non-literal.
template <> struct TupleStorage<std::index_sequence<>> {};

template <std::size_t... I, typename... Values>
struct TupleStorage<std::index_sequence<I...>, Values...>
    : TupleLeaf<Tuple<Values...>, I, Values>... {
  KOKKOS_DEFAULTED_FUNCTION constexpr TupleStorage() = default;

  KOKKOS_INLINE_FUNCTION constexpr explicit TupleStorage(Values... values)
      : TupleLeaf<Tuple<Values...>, I, Values>{ values }... {}
};

// Flat product of device-callable values that preserves constant types.
template <typename... Values>
struct Tuple : TupleStorage<std::index_sequence_for<Values...>, Values...> {
  using TupleStorage<std::index_sequence_for<Values...>,
                     Values...>::TupleStorage;
};

template <typename T> struct TupleTraits {
  static constexpr bool is_tuple = false;
};
template <typename... Values> struct TupleTraits<Tuple<Values...>> {
  static constexpr bool is_tuple = true;
  static constexpr std::size_t rank = sizeof...(Values);
};

// Index sequence over the immediate children of a tuple type.
template <typename T>
inline constexpr auto tuple_indices =
    std::make_index_sequence<TupleTraits<T>::rank>{};

// Overload resolution selects the single base TupleLeaf<Owner, I, T> and
// deduces T, so no type-indexing helper is needed.
template <std::size_t I, typename Owner, typename T, bool Empty>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr T
leaf_get([[maybe_unused]] const TupleLeaf<Owner, I, T, Empty> &leaf) {
  if constexpr (Empty) {
    return T{};
  } else {
    return leaf.value;
  }
}

template <std::size_t I, typename... Values>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tuple_get(const Tuple<Values...> &tuple) {
  return leaf_get<I, Tuple<Values...>>(tuple);
}

template <std::size_t I, typename Tuple>
using tuple_element_t = decltype(tuple_get<I>(std::declval<const Tuple &>()));

template <typename... Values>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tuple_make(Values... values) {
  if constexpr (sizeof...(Values) == 0) {
    return Tuple<>{};
  } else {
    return Tuple<Values...>(values...);
  }
}

template <typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr Head
tuple_head(const Tuple<Head, Tail...> &tuple) {
  return tuple_get<0>(tuple);
}

template <typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tuple_tail(const Tuple<Head, Tail...> &tuple) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return tuple_make(tuple_get<I + 1>(tuple)...);
  }(std::index_sequence_for<Tail...>{});
}

template <typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tuple_prepend(Head head, const Tuple<Tail...> &tail) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return tuple_make(head, tuple_get<I>(tail)...);
  }(std::index_sequence_for<Tail...>{});
}

// Subview selectors are tuple elements too. subview.hpp defines them and
// specializes this trait next to their definitions.
template <typename T> inline constexpr bool is_selector_v = false;

template <typename T> struct StaticTree {
  static constexpr bool value = Constant<T>;
};
template <typename... Values> struct StaticTree<Tuple<Values...>> {
  static constexpr bool value = (StaticTree<Values>::value && ...);
};

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Integral constant leaf for shapes, strides, and coordinates.
 * @tparam Value Compile-time integral value.
 */
template <auto Value>
inline constexpr std::integral_constant<decltype(Value), Value> c{};

/**
 * @brief Whether a type is metadata constructed by make_tuple.
 * @tparam T Metadata type, possibly const or a reference.
 */
template <typename T>
inline constexpr bool is_tuple_v =
    layout_impl::TupleTraits<std::remove_cvref_t<T>>::is_tuple;

/**
 * @brief Number of immediate children in a metadata tuple.
 * @tparam Tuple Metadata tuple type, possibly const or a reference.
 */
template <typename Tuple>
  requires is_tuple_v<Tuple>
inline constexpr std::size_t tuple_size_v =
    layout_impl::TupleTraits<std::remove_cvref_t<Tuple>>::rank;

/**
 * @brief Type accepted as a make_tuple element: a non-boolean integer, an
 * integral constant of one, a nested tuple, or a subview selector (All or
 * Range).
 *
 * Each use checks its own leaves: shapes, strides, and coordinates reject
 * selectors, and subview selectors accept them.
 * @tparam T Candidate element type.
 */
template <typename T>
concept TupleElement = layout_impl::Index<T> || layout_impl::Constant<T> ||
                       is_tuple_v<T> || layout_impl::is_selector_v<T>;

/**
 * @brief Construct a recursive, device-callable tuple without erasing
 * constants.
 *
 * Elements that are empty types, such as integral constants or fully static
 * subtuples, occupy no storage.
 * @tparam Values Types of leaves or nested tuples.
 * @param values Leaves or subtrees in first-leaf-fastest order.
 * @return Hierarchical tuple, including an empty tuple for rank zero.
 */
template <TupleElement... Values>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_tuple(Values... values) {
  return layout_impl::tuple_make(values...);
}

/**
 * @brief Inspect a hierarchical metadata tuple.
 * @tparam I Child position at this level.
 * @tparam Tuple Hierarchical tuple type.
 * @param tuple Tuple whose child is requested.
 * @return Copy of the child, retaining its static type.
 */
template <std::size_t I, typename Tuple>
  requires is_tuple_v<Tuple>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto get(const Tuple &tuple) {
  static_assert(I < tuple_size_v<Tuple>, "layout tuple index out of range");
  return layout_impl::tuple_get<I>(tuple);
}

} // namespace specfem::experimental::layout
