#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>

/** @cond */
namespace specfem::experimental::layout_impl {

// This product stores only device-callable values and preserves constant types.
template <typename... Values> struct Tuple;
template <> struct Tuple<> {};
template <typename Head, typename... Tail> struct Tuple<Head, Tail...> {
  [[no_unique_address]] Head head{};
  [[no_unique_address]] Tuple<Tail...> tail{};
};

template <std::size_t I, typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr const auto &
tuple_get(const Tuple<Head, Tail...> &tuple) {
  if constexpr (I == 0) {
    return tuple.head;
  } else {
    return tuple_get<I - 1>(tuple.tail);
  }
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto tuple_make() {
  return Tuple<>{};
}

template <typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto tuple_make(Head head,
                                                               Tail... tail) {
  return Tuple<Head, Tail...>{ head, tuple_make(tail...) };
}

template <typename Head, typename... Tail>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tuple_prepend(Head head, Tuple<Tail...> tail) {
  return Tuple<Head, Tail...>{ head, tail };
}

template <typename T> struct TupleTraits {
  static constexpr bool is_tuple = false;
};
template <typename... Values> struct TupleTraits<Tuple<Values...>> {
  static constexpr bool is_tuple = true;
  static constexpr std::size_t rank = sizeof...(Values);
};

template <typename T> struct ConstantTraits {
  static constexpr bool is_constant = false;
};
template <typename T, T Value>
struct ConstantTraits<std::integral_constant<T, Value>> {
  static constexpr bool is_constant = true;
};

template <typename T> struct StaticTree : ConstantTraits<T> {
  static constexpr bool value = ConstantTraits<T>::is_constant;
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
 * @brief Construct a recursive, device-callable tuple without erasing
 * constants.
 * @tparam Values Types of leaves or nested tuples.
 * @param values Leaves or subtrees in first-leaf-fastest order.
 * @return Hierarchical tuple, including an empty tuple for rank zero.
 */
template <typename... Values>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_tuple(Values... values) {
  return layout_impl::tuple_make(values...);
}

/**
 * @brief Inspect a hierarchical metadata tuple.
 * @tparam I Child position at this level.
 * @tparam Tuple Hierarchical tuple type.
 * @param tuple Tuple whose child is requested.
 * @return Read-only reference to the child, retaining its static type.
 */
template <std::size_t I, typename Tuple>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr const auto &
get(const Tuple &tuple) {
  static_assert(I < layout_impl::TupleTraits<Tuple>::rank,
                "layout tuple index out of range");
  return layout_impl::tuple_get<I>(tuple);
}

} // namespace specfem::experimental::layout
