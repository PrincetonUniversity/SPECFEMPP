#pragma once

#include "specfem/experimental/layout/tuple.hpp"
#include <limits>

/** @cond */
namespace specfem::experimental::layout_impl {

KOKKOS_INLINE_FUNCTION constexpr void require(bool condition,
                                              const char *message) {
  if (std::is_constant_evaluated()) {
    if (!condition) {
      Kokkos::abort(message);
    }
  }
#ifndef NDEBUG
  if (!condition) {
    Kokkos::abort(message);
  }
#endif
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t add(std::size_t a,
                                                               std::size_t b) {
  require(b <= std::numeric_limits<std::size_t>::max() - a,
          "layout addition overflow");
  return a + b;
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
multiply(std::size_t a, std::size_t b) {
  require(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a,
          "layout multiplication overflow");
  return a * b;
}

template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t integer(T value) {
  if constexpr (Constant<T>) {
    using value_type = typename T::value_type;
    static_assert(T::value >= 0, "layout leaf must be nonnegative");
    static_assert(sizeof(value_type) <= sizeof(std::size_t) ||
                      T::value <= std::numeric_limits<std::size_t>::max(),
                  "layout leaf exceeds index range");
    return static_cast<std::size_t>(T::value);
  } else {
    static_assert(Index<T>, "layout leaves must be non-boolean integers");
    if constexpr (std::is_signed_v<T>) {
      require(value >= 0, "layout leaf must be nonnegative");
    }
    if constexpr (sizeof(T) > sizeof(std::size_t)) {
      require(value <= std::numeric_limits<std::size_t>::max(),
              "layout leaf exceeds index range");
    }
    return static_cast<std::size_t>(value);
  }
}

template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool has_zero(const T &tree) {
  if constexpr (TupleTraits<T>::is_tuple) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (has_zero(tuple_get<I>(tree)) || ...);
    }(tuple_indices<T>);
  } else {
    return integer(tree) == 0;
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr void validate_leaves(const T &tree) {
  if constexpr (TupleTraits<T>::is_tuple) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (validate_leaves(tuple_get<I>(tree)), ...);
    }(tuple_indices<T>);
  } else {
    [[maybe_unused]] const auto checked_value = integer(tree);
  }
}

// Factors are nonzero, so partial products never decrease and the overflow
// check fires exactly when the full product overflows.
template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
product_nonzero(const T &tree) {
  if constexpr (TupleTraits<T>::is_tuple) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      std::size_t result = 1;
      ((result = multiply(result, product_nonzero(tuple_get<I>(tree)))), ...);
      return result;
    }(tuple_indices<T>);
  } else {
    return integer(tree);
  }
}

template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
product(const T &tree) {
  return has_zero(tree) ? 0 : product_nonzero(tree);
}

template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto extent(const T &tree) {
  if constexpr (StaticTree<T>::value) {
    return std::integral_constant<std::size_t, product(T{})>{};
  } else {
    return product(tree);
  }
}

template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool matching_topology() {
  if constexpr (TupleTraits<Shape>::is_tuple != TupleTraits<Stride>::is_tuple) {
    return false;
  } else if constexpr (!TupleTraits<Shape>::is_tuple) {
    return true;
  } else if constexpr (TupleTraits<Shape>::rank != TupleTraits<Stride>::rank) {
    return false;
  } else {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return (matching_topology<tuple_element_t<I, Shape>,
                                tuple_element_t<I, Stride>>() &&
              ...);
    }(tuple_indices<Shape>);
  }
}

// Terms are nonnegative, so the running sum overflows exactly when the total
// does.
template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
last_offset(const Shape &shape, const Stride &stride) {
  if constexpr (TupleTraits<Shape>::is_tuple) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      std::size_t result = 0;
      ((result = add(result,
                     last_offset(tuple_get<I>(shape), tuple_get<I>(stride)))),
       ...);
      return result;
    }(tuple_indices<Shape>);
  } else {
    return multiply(integer(shape) - 1, integer(stride));
  }
}

template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
span(const Shape &shape, const Stride &stride) {
  return has_zero(shape) ? 0 : add(last_offset(shape, stride), 1);
}

// Coordinate bounds are a query, so negative coordinates return false.
template <typename Coordinate>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool
in_bounds(Coordinate coordinate, std::size_t bound) {
  if constexpr (Constant<Coordinate>) {
    return in_bounds(Coordinate::value, bound);
  } else {
    static_assert(Index<Coordinate>,
                  "layout coordinates must be non-boolean integers");
    if constexpr (std::is_signed_v<Coordinate>) {
      if (coordinate < 0) {
        return false;
      }
    }
    return static_cast<std::make_unsigned_t<Coordinate>>(coordinate) < bound;
  }
}

template <typename Shape, typename Coordinate>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool
bounds(const Shape &shape, const Coordinate &coordinate) {
  if constexpr (TupleTraits<Coordinate>::is_tuple) {
    static_assert(TupleTraits<Shape>::is_tuple,
                  "layout coordinate topology must match shape");
    if constexpr (TupleTraits<Shape>::is_tuple) {
      static_assert(TupleTraits<Shape>::rank == TupleTraits<Coordinate>::rank,
                    "layout coordinate rank must match shape");
      if constexpr (TupleTraits<Shape>::rank == TupleTraits<Coordinate>::rank) {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
          return (bounds(tuple_get<I>(shape), tuple_get<I>(coordinate)) && ...);
        }(tuple_indices<Shape>);
      }
    }
  } else {
    return in_bounds(coordinate, product(shape));
  }
}

// Reject decisive constant bounds even when other modes contain runtime values.
template <typename Shape, typename Coordinate>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool constant_bounds() {
  if constexpr (Constant<Coordinate>) {
    if constexpr (Coordinate::value < 0) {
      return false;
    }
  }
  if constexpr (StaticTree<Shape>::value && StaticTree<Coordinate>::value) {
    return bounds(Shape{}, Coordinate{});
  } else if constexpr (TupleTraits<Shape>::is_tuple &&
                       TupleTraits<Coordinate>::is_tuple) {
    if constexpr (TupleTraits<Shape>::rank == TupleTraits<Coordinate>::rank) {
      return []<std::size_t... I>(std::index_sequence<I...>) {
        return (constant_bounds<tuple_element_t<I, Shape>,
                                tuple_element_t<I, Coordinate>>() &&
                ...);
      }(tuple_indices<Shape>);
    }
  }
  return true;
}

template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
scalar_offset(const Shape &shape, const Stride &stride, std::size_t index);

// Decompose index over children I and later, first child fastest.
template <std::size_t I, typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
scalar_offset_from(const Shape &shape, const Stride &stride,
                   std::size_t index) {
  constexpr std::size_t rank = TupleTraits<Shape>::rank;
  if constexpr (I == rank) {
    return 0;
  } else if constexpr (I + 1 == rank) {
    // Bounds already constrain the final child. Another modulo would add a
    // runtime division when that last extent is dynamic.
    return scalar_offset(tuple_get<I>(shape), tuple_get<I>(stride), index);
  } else {
    const auto extent = product(tuple_get<I>(shape));
    return scalar_offset(tuple_get<I>(shape), tuple_get<I>(stride),
                         index % extent) +
           scalar_offset_from<I + 1>(shape, stride, index / extent);
  }
}

// The caller has already established bounds and a nonempty domain.
template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
scalar_offset(const Shape &shape, const Stride &stride, std::size_t index) {
  if constexpr (TupleTraits<Shape>::is_tuple) {
    return scalar_offset_from<0>(shape, stride, index);
  } else {
    return index * integer(stride);
  }
}

template <typename Shape, typename Stride, typename Coordinate>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
coordinate_offset(const Shape &shape, const Stride &stride,
                  const Coordinate &coordinate) {
  if constexpr (TupleTraits<Coordinate>::is_tuple) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (std::size_t{ 0 } + ... +
              coordinate_offset(tuple_get<I>(shape), tuple_get<I>(stride),
                                tuple_get<I>(coordinate)));
    }(tuple_indices<Coordinate>);
  } else {
    return scalar_offset(shape, stride, integer(coordinate));
  }
}

// Dynamic nonnegative strides contribute at least zero to a static span bound.
template <typename T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
minimum_strides(const T &tree) {
  if constexpr (TupleTraits<T>::is_tuple) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return tuple_make(minimum_strides(tuple_get<I>(tree))...);
    }(tuple_indices<T>);
  } else if constexpr (Constant<T>) {
    return tree;
  } else {
    return std::integral_constant<std::size_t, 0>{};
  }
}

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Recursive shape/stride mapping independent of backing storage.
 *
 * Inputs must describe an injective mapping. Uniqueness certification is added
 * separately. Runtime preconditions are checked only without NDEBUG.
 * @tparam Shape Recursive tuple of nonnegative integral extents.
 * @tparam Strides Matching recursive tuple of nonnegative element strides.
 */
template <typename Shape, typename Strides> class Layout {
  static_assert(layout_impl::TupleTraits<Shape>::is_tuple &&
                    layout_impl::TupleTraits<Strides>::is_tuple,
                "layout shape and strides must be tuples");
  static_assert(layout_impl::matching_topology<Shape, Strides>(),
                "layout shape and stride topology must match");

public:
  /**
   * @brief Construct an enabled root mapping with static span diagnostics.
   * @param shape Recursive extents.
   * @param strides Corresponding element strides.
   */
  KOKKOS_INLINE_FUNCTION constexpr Layout(Shape shape, Strides strides)
      : Layout(shape, strides, true) {}

  /**
   * @brief Construct a root mapping with a rectangular logical domain.
   *
   * Type-known span overflow fails compilation whatever the value of enabled,
   * because no runtime flag can make such a layout valid.
   * @param shape Recursive extents.
   * @param strides Corresponding element strides.
   * @param enabled False makes the logical domain empty, including rank zero.
   */
  KOKKOS_INLINE_FUNCTION constexpr Layout(Shape shape, Strides strides,
                                          bool enabled)
      : shape_(shape), strides_(strides), enabled_(enabled) {
    layout_impl::validate_leaves(shape_);
    layout_impl::validate_leaves(strides_);
    if constexpr (layout_impl::StaticTree<Shape>::value) {
      [[maybe_unused]] constexpr auto size = layout_impl::product(Shape{});
      [[maybe_unused]] constexpr auto minimum_span =
          layout_impl::span(Shape{}, layout_impl::minimum_strides(Strides{}));
    }
    if (std::is_constant_evaluated()) {
      [[maybe_unused]] const auto checked_envelope_size = envelope_size();
      [[maybe_unused]] const auto checked_span_size = required_span_size();
    }
#ifndef NDEBUG
    [[maybe_unused]] const auto checked_envelope_size = envelope_size();
    [[maybe_unused]] const auto checked_span_size = required_span_size();
#endif
  }

  /** @brief Number of top-level modes. @return Coordinate rank. */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr std::size_t rank() {
    return layout_impl::TupleTraits<Shape>::rank;
  }

  /** @brief Recursive extent metadata. @return Shape with static leaves intact.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr const Shape &shape() const {
    return shape_;
  }

  /** @brief Recursive element strides. @return Strides with static leaves
   * intact. */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr const Strides &
  strides() const {
    return strides_;
  }

  /**
   * @brief Scalar extent of a top-level mode.
   * @tparam I Mode position.
   * @return Product of its leaf extents, as
   * `std::integral_constant<std::size_t, N>` if every leaf of the mode is
   * static, otherwise `std::size_t`. Store the result in `std::size_t` to
   * modify it.
   */
  template <std::size_t I>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto extent() const {
    return layout_impl::extent(get<I>(shape_));
  }

  /**
   * @brief Size of the rectangular coordinate envelope.
   * @return Product of all shape leaves, as
   * `std::integral_constant<std::size_t, N>` if the shape is fully static,
   * otherwise `std::size_t`.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto envelope_size() const {
    return layout_impl::extent(shape_);
  }

  /** @brief Whether the logical domain is empty. @return True if no access is
   * valid. */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool empty() const {
    return !enabled_ || layout_impl::has_zero(shape_);
  }

  /**
   * @brief Required capacity measured from the original backing pointer.
   * @return Zero for empty domains, otherwise maximum element offset plus one.
   * Always `std::size_t`, because it depends on the runtime enabled flag.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
  required_span_size() const {
    return empty() ? 0 : layout_impl::span(shape_, strides_);
  }

  /**
   * @brief Test logical membership without accessing storage.
   * @tparam Coordinate Tuple following shape topology, with scalar subtrees.
   * @param coordinate Logical coordinate to test.
   * @return False for empty domains or out-of-bounds coordinates.
   */
  template <typename Coordinate>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool
  valid(const Coordinate &coordinate) const {
    static_assert(layout_impl::TupleTraits<Coordinate>::is_tuple,
                  "layout coordinates must be tuples at the top level");
    return !empty() && layout_impl::bounds(shape_, coordinate);
  }

  /**
   * @brief Map a valid coordinate to an element offset.
   * @tparam Coordinate Tuple following shape topology, with scalar subtrees.
   * @param coordinate Valid logical coordinate, checked in debug builds.
   * @return Element offset relative to the original backing pointer, as
   * `std::integral_constant<std::size_t, N>` if the shape, strides, and
   * coordinate are all static, otherwise `std::size_t`.
   */
  template <typename Coordinate>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
  offset(const Coordinate &coordinate) const {
    static_assert(layout_impl::constant_bounds<Shape, Coordinate>(),
                  "layout constant coordinate out of bounds");
    layout_impl::require(valid(coordinate), "layout coordinate out of bounds");
    if constexpr (layout_impl::StaticTree<Shape>::value &&
                  layout_impl::StaticTree<Strides>::value &&
                  layout_impl::StaticTree<Coordinate>::value) {
      return std::integral_constant<std::size_t,
                                    layout_impl::coordinate_offset(
                                        Shape{}, Strides{}, Coordinate{})>{};
    } else {
      return layout_impl::coordinate_offset(shape_, strides_, coordinate);
    }
  }

private:
  Shape shape_{};
  Strides strides_{};
  bool enabled_ = true;
};

/**
 * @brief Deduce a standalone recursive layout.
 * @tparam Shape Recursive shape tuple.
 * @tparam Strides Matching recursive stride tuple.
 * @param shape Extents, with first-leaf-fastest scalar interpretation.
 * @param strides Element strides, including any padding.
 * @return Enabled mapping with type-known span diagnostics.
 */
template <typename Shape, typename Strides>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_layout(Shape shape, Strides strides) {
  return Layout<Shape, Strides>(shape, strides);
}

/**
 * @brief Deduce a layout whose logical domain can be disabled at runtime.
 * @tparam Shape Recursive shape tuple.
 * @tparam Strides Matching recursive stride tuple.
 * @param shape Extents, with first-leaf-fastest scalar interpretation.
 * @param strides Element strides, including any padding.
 * @param enabled False disables the domain and its backing span requirements.
 * @return Mapping with type-known span diagnostics, whose runtime span checks
 * respect enabled.
 */
template <typename Shape, typename Strides>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_layout(Shape shape, Strides strides, bool enabled) {
  return Layout<Shape, Strides>(shape, strides, enabled);
}

} // namespace specfem::experimental::layout
