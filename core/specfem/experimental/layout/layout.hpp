#pragma once

#include "specfem/experimental/layout/tuple.hpp"
#include <concepts>
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

// Runtime checks run during constant evaluation and in debug builds only, so
// release builds compile them out entirely.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool checks_enabled() {
  if (std::is_constant_evaluated()) {
    return true;
  }
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

template <typename T> struct LeafCount {
  static constexpr std::size_t value = 1;
};
template <typename... Values> struct LeafCount<Tuple<Values...>> {
  static constexpr std::size_t value =
      (std::size_t{ 0 } + ... + LeafCount<Values>::value);
};

struct Leaf {
  std::size_t extent = 0;
  std::size_t stride = 0;
};

// Fixed-capacity leaf list usable in constant evaluation and device code.
template <std::size_t Capacity> struct Leaves {
  Leaf leaf[Capacity == 0 ? 1 : Capacity]{};
  std::size_t size = 0;
};

// Collect leaves with extent at least two; singletons cannot collide.
template <typename Shape, typename Stride, std::size_t Capacity>
KOKKOS_INLINE_FUNCTION constexpr void collect_leaves(const Shape &shape,
                                                     const Stride &stride,
                                                     Leaves<Capacity> &leaves) {
  if constexpr (TupleTraits<Shape>::is_tuple) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (collect_leaves(tuple_get<I>(shape), tuple_get<I>(stride), leaves), ...);
    }(tuple_indices<Shape>);
  } else {
    const auto extent = integer(shape);
    if (extent > 1) {
      leaves.leaf[leaves.size] = Leaf{ extent, integer(stride) };
      ++leaves.size;
    }
  }
}

enum class Uniqueness { certified, broadcast, overlap };

// Sufficient injectivity proof over the rectangular envelope. After sorting
// leaves by stride, each stride must exceed the largest offset reachable by
// the leaves before it. Two distinct coordinates then differ at a highest
// sorted leaf whose stride exceeds every lower contribution, so their offsets
// differ. Some injective layouts, such as (2, 3) : (3, 2), are rejected.
template <typename Shape, typename Stride>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr Uniqueness
certify(const Shape &shape, const Stride &stride) {
  if (has_zero(shape)) {
    return Uniqueness::certified;
  }
  Leaves<LeafCount<Shape>::value> leaves{};
  collect_leaves(shape, stride, leaves);
  for (std::size_t i = 1; i < leaves.size; ++i) {
    const Leaf current = leaves.leaf[i];
    std::size_t j = i;
    for (; j > 0 && leaves.leaf[j - 1].stride > current.stride; --j) {
      leaves.leaf[j] = leaves.leaf[j - 1];
    }
    leaves.leaf[j] = current;
  }
  std::size_t covered = 0;
  for (std::size_t i = 0; i < leaves.size; ++i) {
    const Leaf leaf = leaves.leaf[i];
    if (leaf.stride == 0) {
      return Uniqueness::broadcast;
    }
    if (leaf.stride <= covered) {
      return Uniqueness::overlap;
    }
    covered = add(covered, multiply(leaf.extent - 1, leaf.stride));
  }
  return Uniqueness::certified;
}

template <typename Shape, typename Stride>
KOKKOS_INLINE_FUNCTION constexpr void require_unique(const Shape &shape,
                                                     const Stride &stride) {
  const auto uniqueness = certify(shape, stride);
  require(uniqueness != Uniqueness::broadcast,
          "layout broadcast: zero stride on a nonsingleton leaf");
  require(uniqueness != Uniqueness::overlap,
          "layout uniqueness not certified: stride separation failed");
}

// Add a displacement to an offset or span. A static zero displacement adds no
// instruction, so root layouts keep their original address arithmetic.
template <typename Displacement>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
displace(Displacement displacement, std::size_t value) {
  if constexpr (Constant<Displacement>) {
    if constexpr (Displacement::value == 0) {
      return value;
    } else {
      return add(integer(displacement), value);
    }
  } else {
    return add(integer(displacement), value);
  }
}

using static_zero = std::integral_constant<std::size_t, 0>;

template <typename T>
inline constexpr bool is_displacement_v = std::is_same_v<T, std::size_t>;
template <std::size_t Value>
inline constexpr bool
    is_displacement_v<std::integral_constant<std::size_t, Value>> = true;

// Selects the layout constructor that skips uniqueness certification.
struct DeriveTag {};

// First-leaf-fastest scalar index of an in-bounds coordinate.
template <typename Shape, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
linearize(const Shape &shape, const Coordinate &coordinate) {
  if constexpr (!TupleTraits<Coordinate>::is_tuple) {
    return integer(coordinate);
  } else if constexpr (TupleTraits<Shape>::rank == 0) {
    return 0;
  } else {
    return linearize(tuple_head(shape), tuple_head(coordinate)) +
           product(tuple_head(shape)) *
               linearize(tuple_tail(shape), tuple_tail(coordinate));
  }
}

template <typename Shape, typename Bounds, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr bool
logical_contains(const Shape &shape, const Bounds &bounds,
                 const Coordinate &coord) {
  if constexpr (TupleTraits<Bounds>::rank == 0) {
    return true;
  } else {
    return linearize(tuple_head(shape), tuple_head(coord)) <
               integer(tuple_head(bounds)) &&
           logical_contains(tuple_tail(shape), tuple_tail(bounds),
                            tuple_tail(coord));
  }
}

// A constant scalar bound can reject one mode independently of other modes.
template <typename Shape, typename Bound, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr bool constant_logical_mode() {
  if constexpr (Constant<Bound>) {
    if constexpr (Constant<Coordinate>) {
      return in_bounds(Coordinate{}, integer(Bound{}));
    } else if constexpr (StaticTree<Shape>::value &&
                         StaticTree<Coordinate>::value) {
      return linearize(Shape{}, Coordinate{}) < integer(Bound{});
    }
  }
  return true;
}

template <typename Shape, typename Bounds, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr bool constant_logical_bounds() {
  if constexpr (TupleTraits<Bounds>::rank != 0 &&
                TupleTraits<Coordinate>::is_tuple) {
    if constexpr (TupleTraits<Bounds>::rank == TupleTraits<Coordinate>::rank) {
      return constant_logical_mode<tuple_element_t<0, Shape>,
                                   tuple_element_t<0, Bounds>,
                                   tuple_element_t<0, Coordinate>>() &&
             constant_logical_bounds<decltype(tuple_tail(Shape{})),
                                     decltype(tuple_tail(Bounds{})),
                                     decltype(tuple_tail(Coordinate{}))>();
    }
  }
  return true;
}

template <typename Shape, typename Bounds>
KOKKOS_INLINE_FUNCTION constexpr void
check_logical_bounds(const Shape &shape, const Bounds &bounds) {
  if constexpr (TupleTraits<Bounds>::rank != 0) {
    require(integer(tuple_head(bounds)) <= product(tuple_head(shape)),
            "layout logical bound exceeds envelope");
    check_logical_bounds(tuple_tail(shape), tuple_tail(bounds));
  }
}

template <std::size_t I = 0, typename M>
KOKKOS_INLINE_FUNCTION constexpr std::size_t runtime_extent(const M &mapping,
                                                            std::size_t axis) {
  if constexpr (I == M::rank()) {
    require(false, "layout extent axis out of bounds");
    return 0;
  } else {
    return axis == I ? integer(mapping.template extent<I>())
                     : runtime_extent<I + 1>(mapping, axis);
  }
}

struct LayoutAccess;

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Recursive shape/stride mapping independent of backing storage.
 *
 * Construction certifies that distinct valid coordinates map to distinct
 * element offsets. Leaves with extent one are ignored. The remaining leaves,
 * sorted by stride, must each have a stride larger than the largest offset
 * reachable by the leaves before it. This rule is sufficient, not necessary:
 * it rejects some injective layouts, such as shape (2, 3) with strides (3, 2),
 * and it never accepts a broadcast or an overlap. Empty domains have no valid
 * coordinate and are always certified. Unused padding between offsets is
 * allowed.
 *
 * Fully static, default-enabled descriptors that fail certification do not
 * compile. Other descriptors are checked during constant evaluation and in
 * builds without NDEBUG, on host and device. In release builds an uncertified
 * descriptor is a precondition violation with undefined behavior.
 *
 * Every offset is the displacement plus the sum of coordinate leaves times
 * their strides. Root layouts have a static zero displacement. Subviews add a
 * displacement so that offsets stay relative to the original backing pointer.
 * @tparam Shape Recursive tuple of nonnegative integral extents.
 * @tparam Strides Matching recursive tuple of nonnegative element strides.
 * @tparam Displacement Offset of coordinate zero, either a std::size_t or a
 * std::size_t integral constant.
 * @tparam LogicalBounds Optional tuple of top-level logical scalar bounds.
 */
template <typename Shape, typename Strides,
          typename Displacement = layout_impl::static_zero,
          typename LogicalBounds = void>
class Layout {
  static_assert(layout_impl::TupleTraits<Shape>::is_tuple &&
                    layout_impl::TupleTraits<Strides>::is_tuple,
                "layout shape and strides must be tuples");
  static_assert(layout_impl::matching_topology<Shape, Strides>(),
                "layout shape and stride topology must match");

public:
  using shape_type = Shape;
  using strides_type = Strides;
  using logical_bounds_type = LogicalBounds;
  static constexpr bool is_transformed = false;

  /** @brief Construct an uninitialized descriptor for an empty view handle. */
  KOKKOS_INLINE_FUNCTION constexpr Layout()
      : Layout(layout_impl::DeriveTag{}, Shape{}, Strides{}, false) {
    if constexpr (!std::is_void_v<LogicalBounds>) {
      layout_impl::validate_leaves(logical_);
    }
  }

  /**
   * @brief Restrict a root envelope by top-level logical scalar bounds.
   * Type-known span overflow and uniqueness failures do not compile, including
   * with empty logical bounds.
   * @tparam Bounds Tuple matching the logical bound metadata type.
   * @param shape Rectangular coordinate envelope.
   * @param strides Corresponding element strides.
   * @param logical Logical extents excluding padding.
   */
  template <typename Bounds>
  KOKKOS_INLINE_FUNCTION constexpr Layout(Shape shape, Strides strides,
                                          Bounds logical)
    requires std::is_same_v<Bounds, LogicalBounds>
      : Layout(layout_impl::DeriveTag{}, shape, strides,
               !layout_impl::has_zero(logical)) {
    static_assert(layout_impl::TupleTraits<Bounds>::rank == rank(),
                  "layout logical bound rank must match shape");
    logical_ = logical;
    layout_impl::validate_leaves(logical_);
    layout_impl::check_logical_bounds(shape_, logical_);
    check_root_descriptor();
  }

  /**
   * @brief Construct an enabled root mapping with static diagnostics.
   *
   * Fully static descriptors that overflow, broadcast, or fail uniqueness
   * certification do not compile.
   * @param shape Recursive extents.
   * @param strides Corresponding element strides.
   */
  KOKKOS_INLINE_FUNCTION constexpr Layout(Shape shape, Strides strides)
    requires std::is_void_v<LogicalBounds>
      : Layout(layout_impl::DeriveTag{}, shape, strides, true) {
    check_root_descriptor();
  }

  /**
   * @brief Construct a root mapping with a rectangular logical domain.
   *
   * Type-known span overflow and uniqueness failures do not compile, whatever
   * the value of enabled, because no runtime flag can make such a layout
   * valid. Other nonempty domains are certified unique during constant
   * evaluation and in builds without NDEBUG.
   * @param shape Recursive extents.
   * @param strides Corresponding element strides.
   * @param enabled False makes the logical domain empty, including rank zero.
   */
  KOKKOS_INLINE_FUNCTION constexpr Layout(Shape shape, Strides strides,
                                          bool enabled)
    requires std::is_void_v<LogicalBounds>
      : Layout(layout_impl::DeriveTag{}, shape, strides, enabled) {
    check_root_descriptor();
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
   * @brief Offset of coordinate zero relative to the original backing pointer.
   * @return Static zero for root layouts. Subviews add the selected start
   * offsets; the result is static when every added term is static.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr Displacement
  displacement() const {
    return displacement_;
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
    if constexpr (std::is_void_v<LogicalBounds>) {
      return layout_impl::extent(get<I>(shape_));
    } else {
      return get<I>(logical_);
    }
  }

  /** @brief Inspect a logical extent by axis. @param axis Top-level axis.
   * @return Logical scalar extent. */
  KOKKOS_INLINE_FUNCTION constexpr std::size_t extent(std::size_t axis) const {
    return layout_impl::runtime_extent(*this, axis);
  }

  /** @brief Full physical envelope span. @return Original allocation span. */
  KOKKOS_INLINE_FUNCTION constexpr std::size_t physical_span_size() const {
    return required_span_size();
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
   * @return Zero for empty domains, otherwise a safe upper backing bound.
   * Always `std::size_t`, because it depends on the runtime enabled flag.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t
  required_span_size() const {
    return empty() ? 0
                   : layout_impl::displace(displacement_,
                                           layout_impl::span(shape_, strides_));
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
    if (empty() || !layout_impl::bounds(shape_, coordinate)) {
      return false;
    }
    if constexpr (std::is_void_v<LogicalBounds>) {
      return true;
    } else {
      return layout_impl::logical_contains(shape_, logical_, coordinate);
    }
  }

  /**
   * @brief Map a valid coordinate to an element offset.
   * @tparam Coordinate Tuple following shape topology, with scalar subtrees.
   * @param coordinate Valid logical coordinate, checked in debug builds.
   * @return Element offset relative to the original backing pointer, as
   * `std::integral_constant<std::size_t, N>` if the shape, strides,
   * displacement, and coordinate are all static, otherwise `std::size_t`.
   */
  template <typename Coordinate>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
  offset(const Coordinate &coordinate) const {
    static_assert(layout_impl::constant_bounds<Shape, Coordinate>(),
                  "layout constant coordinate out of bounds");
    if constexpr (!std::is_void_v<LogicalBounds>) {
      static_assert(layout_impl::constant_logical_bounds<Shape, LogicalBounds,
                                                         Coordinate>(),
                    "layout constant coordinate outside logical bounds");
    }
    layout_impl::require(valid(coordinate), "layout coordinate out of bounds");
    if constexpr (layout_impl::StaticTree<Shape>::value &&
                  layout_impl::StaticTree<Strides>::value &&
                  layout_impl::StaticTree<Coordinate>::value &&
                  layout_impl::Constant<Displacement>) {
      return std::integral_constant<
          std::size_t,
          layout_impl::displace(
              Displacement{}, layout_impl::coordinate_offset(Shape{}, Strides{},
                                                             Coordinate{}))>{};
    } else {
      return layout_impl::displace(
          displacement_,
          layout_impl::coordinate_offset(shape_, strides_, coordinate));
    }
  }

private:
  friend struct layout_impl::LayoutAccess;

  // Transformations construct derived layouts here. Leaf, envelope, and span
  // checks still run, but uniqueness is inherited from the parent instead of
  // re-certified: the envelope of a tiled slice can overlap where its
  // inherited validity cannot.
  KOKKOS_INLINE_FUNCTION constexpr Layout(layout_impl::DeriveTag, Shape shape,
                                          Strides strides, bool enabled,
                                          Displacement displacement = {})
      : shape_(shape), strides_(strides), displacement_(displacement),
        enabled_(enabled) {
    static_assert(layout_impl::is_displacement_v<Displacement>,
                  "layout displacement must be std::size_t or a constant");
    layout_impl::validate_leaves(shape_);
    layout_impl::validate_leaves(strides_);
    if constexpr (layout_impl::StaticTree<Shape>::value) {
      [[maybe_unused]] constexpr auto size = layout_impl::product(Shape{});
    }
    if (layout_impl::checks_enabled()) {
      [[maybe_unused]] const auto checked_envelope_size = envelope_size();
      [[maybe_unused]] const auto checked_span_size = required_span_size();
    }
  }

  // Certify root layouts during constant evaluation and in debug builds.
  KOKKOS_INLINE_FUNCTION constexpr void check_unique() const {
    if (layout_impl::checks_enabled() && !empty()) {
      layout_impl::require_unique(shape_, strides_);
    }
  }

  // Checks shared by every public root constructor. Span overflow and
  // uniqueness that depend only on the types fail compilation, whatever the
  // enabled flag, because no runtime value can make such a layout valid.
  // Other descriptors are certified at runtime by check_unique.
  KOKKOS_INLINE_FUNCTION constexpr void check_root_descriptor() const {
    if constexpr (layout_impl::StaticTree<Shape>::value) {
      [[maybe_unused]] constexpr auto minimum_span =
          layout_impl::span(Shape{}, layout_impl::minimum_strides(Strides{}));
    }
    if constexpr (layout_impl::StaticTree<Shape>::value &&
                  layout_impl::StaticTree<Strides>::value) {
      constexpr auto uniqueness = layout_impl::certify(Shape{}, Strides{});
      static_assert(uniqueness != layout_impl::Uniqueness::broadcast,
                    "layout broadcast: zero stride on a nonsingleton leaf");
      static_assert(
          uniqueness != layout_impl::Uniqueness::overlap,
          "layout uniqueness not certified: stride separation failed");
    } else {
      check_unique();
    }
  }

  [[no_unique_address]] std::conditional_t<std::is_void_v<LogicalBounds>,
                                           layout_impl::Tuple<>, LogicalBounds>
      logical_{};
  Shape shape_{};
  Strides strides_{};
  [[no_unique_address]] Displacement displacement_{};
  bool enabled_ = true;
};

/**
 * @brief Deduce a standalone recursive layout.
 * @tparam Shape Recursive shape tuple.
 * @tparam Strides Matching recursive stride tuple.
 * @param shape Extents, with first-leaf-fastest scalar interpretation.
 * @param strides Element strides, including any padding.
 * @return Enabled mapping, certified unique. Type-known span and uniqueness
 * failures do not compile.
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
 * @return Mapping with type-known span and uniqueness diagnostics, whose
 * runtime span and uniqueness checks apply only when enabled.
 */
template <typename Shape, typename Strides>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_layout(Shape shape, Strides strides, bool enabled) {
  return Layout<Shape, Strides>(shape, strides, enabled);
}

/**
 * @brief Construct a padded root with explicit logical scalar bounds.
 * @tparam Shape Envelope shape type.
 * @tparam Strides Element stride type.
 * @tparam Bounds Logical scalar extent tuple type.
 * @param shape Rectangular envelope.
 * @param strides Corresponding strides.
 * @param logical Logical extents, excluding padding.
 * @return Shared layout retaining logical bounds and physical metadata.
 */
template <typename Shape, typename Strides, typename Bounds>
KOKKOS_INLINE_FUNCTION constexpr auto make_layout(Shape shape, Strides strides,
                                                  Bounds logical)
  requires is_tuple_v<Bounds>
{
  return Layout<Shape, Strides, layout_impl::static_zero, Bounds>(
      shape, strides, logical);
}

} // namespace specfem::experimental::layout

/** @cond */
namespace specfem::experimental::layout_impl {

// The only construction path that skips uniqueness certification. A caller
// must guarantee that every valid derived coordinate corresponds to exactly one
// valid parent coordinate with the same offset, so the parent's certificate
// carries over.
struct LayoutAccess {
  template <typename Shape, typename Strides,
            typename Displacement = static_zero>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr auto
  derive(Shape shape, Strides strides, bool enabled,
         Displacement displacement = {}) {
    return specfem::experimental::layout::Layout<Shape, Strides, Displacement>(
        DeriveTag{}, shape, strides, enabled, displacement);
  }
};

// The protocol every mapping provides to transformations and views. Coordinate
// queries (valid and offset) are templates and are not listed here.
// required_span_size() is zero when empty, otherwise it bounds every valid
// offset plus one. A derived mapping reports at most its parent's span.
// A root Layout reports the displaced rectangular envelope span.
template <typename M>
concept Mapping = requires(const M &mapping) {
  { M::rank() } -> std::convertible_to<std::size_t>;
  mapping.shape();
  mapping.strides();
  mapping.displacement();
  mapping.envelope_size();
  { mapping.empty() } -> std::same_as<bool>;
  { mapping.required_span_size() } -> std::convertible_to<std::size_t>;
};

} // namespace specfem::experimental::layout_impl
/** @endcond */
