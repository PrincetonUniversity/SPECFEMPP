#pragma once

#include "specfem/experimental/layout/transform.hpp"
#include <utility>

namespace specfem::experimental::layout {

/** @brief Selector type that keeps a whole mode. */
struct All {};

/** @brief Selector that keeps a whole mode, like Kokkos::ALL. */
inline constexpr All all{};

/**
 * @brief Selector that keeps the half-open range [begin, end) of a mode.
 * @tparam Begin Non-boolean integer or integral constant type.
 * @tparam End Non-boolean integer or integral constant type.
 */
template <typename Begin, typename End> struct Range {
  Begin begin{}; ///< First selected scalar index.
  End end{};     ///< One past the last selected scalar index.
};

} // namespace specfem::experimental::layout

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename T> inline constexpr bool is_bool_constant_v = false;
template <bool Value>
inline constexpr bool is_bool_constant_v<std::integral_constant<bool, Value>> =
    true;

// Indices and range bounds: non-boolean integers or integer constants.
template <typename T>
inline constexpr bool is_index_v = Index<T> || Constant<T>;

template <typename T>
inline constexpr bool is_all_v = std::is_same_v<T, layout::All>;

template <typename T> inline constexpr bool is_range_v = false;
template <typename Begin, typename End>
inline constexpr bool is_range_v<layout::Range<Begin, End>> = true;

template <> inline constexpr bool is_selector_v<layout::All> = true;
template <typename Begin, typename End>
inline constexpr bool is_selector_v<layout::Range<Begin, End>> = true;

} // namespace specfem::experimental::layout_impl
/** @endcond */

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr bool nonnegative(T value) {
  if constexpr (Constant<T>) {
    return T::value >= 0;
  } else if constexpr (std::is_signed_v<T>) {
    return value >= 0;
  } else {
    return true;
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr std::size_t unsigned_value(T value) {
  if constexpr (Constant<T>) {
    return static_cast<std::size_t>(T::value);
  } else {
    return static_cast<std::size_t>(value);
  }
}

// a <= b on runtime parameters. Comparing substituted constants directly makes
// NVCC warn about pointless comparisons when one side is zero.
KOKKOS_INLINE_FUNCTION constexpr bool ordered(std::size_t a, std::size_t b) {
  return a <= b;
}

// begin <= end <= extent, tested without forming a negative or overflowing
// value.
template <typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr bool range_in_bounds(Begin begin, End end,
                                                      std::size_t extent) {
  return nonnegative(begin) && nonnegative(end) &&
         ordered(unsigned_value(begin), unsigned_value(end)) &&
         ordered(unsigned_value(end), extent);
}

template <typename Range>
inline constexpr bool static_range_v =
    Constant<decltype(Range{}.begin)> && Constant<decltype(Range{}.end)>;

// Shape, strides and displacement contributed by one selected node or tuple.
template <typename Shape, typename Strides, typename Displacement>
struct Selected {
  Shape shape{};
  Strides strides{};
  Displacement displacement{};
};

template <typename Shape, typename Strides, typename Displacement>
KOKKOS_INLINE_FUNCTION constexpr Selected<Shape, Strides, Displacement>
make_selected(Shape shape, Strides strides, Displacement displacement) {
  return { shape, strides, displacement };
}

// Static displacements stay static so that subviews keep root arithmetic.
template <typename A, typename B>
KOKKOS_INLINE_FUNCTION constexpr auto sum(A a, B b) {
  if constexpr (Constant<A> && Constant<B>) {
    // Overflow fails constant evaluation with "layout addition overflow".
    constexpr std::size_t total = add(integer(A{}), integer(B{}));
    return std::integral_constant<std::size_t, total>{};
  } else {
    return add(integer(a), integer(b));
  }
}

// When the result is empty, runtime displacement terms are skipped: an empty
// parent needs no representable span, so its offsets may overflow.
template <typename Shape, typename Stride, typename Index>
KOKKOS_INLINE_FUNCTION constexpr auto
fix_node(const Shape &shape, const Stride &stride, Index index, bool enabled) {
  if constexpr (Constant<Index> && StaticTree<Shape>::value) {
    static_assert(in_bounds(Index::value, product(Shape{})),
                  "layout subview index out of range");
  }
  require(in_bounds(index, product(shape)),
          "layout subview index out of range");
  if constexpr (Constant<Index> && StaticTree<Shape>::value &&
                StaticTree<Stride>::value) {
    constexpr std::size_t offset =
        scalar_offset(Shape{}, Stride{}, integer(Index{}));
    return make_selected(Tuple<>{}, Tuple<>{},
                         std::integral_constant<std::size_t, offset>{});
  } else {
    return make_selected(Tuple<>{}, Tuple<>{},
                         enabled ? scalar_offset(shape, stride, integer(index))
                                 : std::size_t{ 0 });
  }
}

template <typename T> inline constexpr bool is_zero_constant_v = false;
template <typename T, T Value>
inline constexpr bool is_zero_constant_v<std::integral_constant<T, Value>> =
    Value == 0;

template <typename Shape, typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr void
check_range(const Shape &shape, const layout::Range<Begin, End> &range) {
  // Repeats the check in range() for a Range built as an aggregate.
  static_assert(is_index_v<Begin> && is_index_v<End>,
                "layout subview range bounds must be integers");
  if constexpr (static_range_v<layout::Range<Begin, End>> &&
                StaticTree<Shape>::value) {
    static_assert(range_in_bounds(Begin{}, End{}, product(Shape{})),
                  "layout subview range out of order or out of range");
  }
  require(range_in_bounds(range.begin, range.end, product(shape)),
          "layout subview range out of order or out of range");
}

// Number of selected scalar indices, static when both bounds are static.
template <typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr auto
range_count(const layout::Range<Begin, End> &range) {
  if constexpr (static_range_v<layout::Range<Begin, End>>) {
    return std::integral_constant<std::size_t, unsigned_value(End{}) -
                                                   unsigned_value(Begin{})>{};
  } else {
    return unsigned_value(range.end) - unsigned_value(range.begin);
  }
}

// Displacement (begin / block) * stride of a range. An empty range or an empty
// result adds none, because the product could overflow there.
template <typename Stride, typename Block, typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr auto
range_displacement(const Stride &stride, const Block &block,
                   const layout::Range<Begin, End> &range, bool enabled) {
  if constexpr (is_zero_constant_v<Begin>) {
    return static_zero{};
  } else if constexpr (static_range_v<layout::Range<Begin, End>> &&
                       Constant<Stride> && Constant<Block>) {
    constexpr std::size_t begin = unsigned_value(Begin{});
    constexpr std::size_t value =
        begin == unsigned_value(End{})
            ? 0
            : multiply(begin / integer(Block{}), integer(Stride{}));
    return std::integral_constant<std::size_t, value>{};
  } else {
    const std::size_t begin = unsigned_value(range.begin);
    return !enabled || begin == unsigned_value(range.end)
               ? std::size_t{ 0 }
               : multiply(begin / integer(block), integer(stride));
  }
}

template <typename Shape, typename Stride, typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr auto
range_leaf(const Shape &shape, const Stride &stride,
           const layout::Range<Begin, End> &range, bool enabled) {
  check_range(shape, range);
  return make_selected(
      layout::make_tuple(range_count(range)), layout::make_tuple(stride),
      range_displacement(stride, std::integral_constant<std::size_t, 1>{},
                         range, enabled));
}

// Index of the last child that contains a leaf. Its last leaf is the slowest
// leaf of the node in first-leaf-fastest order.
template <typename... Children>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
last_leaf_child(const Tuple<Children...> &) {
  constexpr std::size_t counts[] = { LeafCount<Children>::value..., 0 };
  std::size_t index = 0;
  for (std::size_t i = 0; i < sizeof...(Children); ++i) {
    if (counts[i] > 0) {
      index = i;
    }
  }
  return index;
}

template <typename Tree>
KOKKOS_INLINE_FUNCTION constexpr auto last_leaf(const Tree &tree) {
  if constexpr (TupleTraits<Tree>::is_tuple) {
    constexpr std::size_t k = last_leaf_child(Tree{});
    return last_leaf(layout::get<k>(tree));
  } else {
    return tree;
  }
}

template <typename Tree, typename Value>
KOKKOS_INLINE_FUNCTION constexpr auto replace_last_leaf(const Tree &tree,
                                                        const Value &value) {
  if constexpr (TupleTraits<Tree>::is_tuple) {
    constexpr std::size_t k = last_leaf_child(Tree{});
    return replace_child<k>(
        tree, replace_last_leaf(layout::get<k>(tree), value),
        std::make_index_sequence<TupleTraits<Tree>::rank>{});
  } else {
    return value;
  }
}

// ceil(count / block) for count > 0, without adding block before dividing.
template <typename Count, typename Block>
KOKKOS_INLINE_FUNCTION constexpr auto block_count(const Count &count,
                                                  const Block &block) {
  if constexpr (Constant<Count> && Constant<Block>) {
    constexpr std::size_t n = integer(Count{});
    constexpr std::size_t p = integer(Block{});
    return std::integral_constant<std::size_t, n == 0 ? 0 : (n - 1) / p + 1>{};
  } else {
    const std::size_t n = integer(count);
    return n == 0 ? std::size_t{ 0 } : (n - 1) / integer(block) + 1;
  }
}

// A range on a nested node keeps every leaf but the slowest one, L. The start
// must be a multiple of the block P formed by the other leaves. L covers
// ceil(n / P) blocks; validity removes the rounded tail.
template <typename Shape, typename Stride, typename Begin, typename End>
KOKKOS_INLINE_FUNCTION constexpr auto
range_nested(const Shape &shape, const Stride &stride,
             const layout::Range<Begin, End> &range, bool enabled) {
  using range_type = layout::Range<Begin, End>;
  check_range(shape, range);
  if constexpr (LeafCount<Shape>::value == 0) {
    // Extent one: the envelope is unchanged and the count alone carries
    // emptiness.
    return make_selected(layout::make_tuple(shape), layout::make_tuple(stride),
                         static_zero{});
  } else {
    const auto block = layout_impl::extent(
        replace_last_leaf(shape, std::integral_constant<std::size_t, 1>{}));
    using block_type = std::remove_cvref_t<decltype(block)>;
    if constexpr (static_range_v<range_type> && Constant<block_type>) {
      static_assert(unsigned_value(Begin{}) == unsigned_value(End{}) ||
                        unsigned_value(Begin{}) % integer(block_type{}) == 0,
                    "layout subview range start not aligned to nested mode "
                    "blocks");
    }
    const auto count = range_count(range);
    require(integer(count) == 0 ||
                unsigned_value(range.begin) % integer(block) == 0,
            "layout subview range start not aligned to nested mode blocks");
    return make_selected(
        layout::make_tuple(replace_last_leaf(shape, block_count(count, block))),
        layout::make_tuple(stride),
        range_displacement(last_leaf(stride), block, range, enabled));
  }
}

template <typename Shape, typename Stride, typename Selectors>
KOKKOS_INLINE_FUNCTION constexpr auto
select_children(const Shape &shape, const Stride &stride,
                const Selectors &selectors, bool enabled);

template <typename Shape, typename Selectors>
KOKKOS_INLINE_FUNCTION constexpr bool selector_matches() {
  if constexpr (TupleTraits<Shape>::is_tuple) {
    return TupleTraits<Shape>::rank == TupleTraits<Selectors>::rank;
  } else {
    return false;
  }
}

template <typename Shape, typename Stride, typename Selector>
KOKKOS_INLINE_FUNCTION constexpr auto
select_node(const Shape &shape, const Stride &stride, const Selector &selector,
            bool enabled) {
  if constexpr (is_index_v<Selector>) {
    return fix_node(shape, stride, selector, enabled);
  } else if constexpr (is_all_v<Selector>) {
    return make_selected(layout::make_tuple(shape), layout::make_tuple(stride),
                         static_zero{});
  } else if constexpr (is_range_v<Selector>) {
    if constexpr (TupleTraits<Shape>::is_tuple) {
      return range_nested(shape, stride, selector, enabled);
    } else {
      return range_leaf(shape, stride, selector, enabled);
    }
  } else if constexpr (TupleTraits<Selector>::is_tuple) {
    constexpr bool matching = selector_matches<Shape, Selector>();
    static_assert(matching,
                  "layout subview tuple selector must match a nested mode");
    if constexpr (matching) {
      const auto inner = select_children(shape, stride, selector, enabled);
      return make_selected(layout::make_tuple(inner.shape),
                           layout::make_tuple(inner.strides),
                           inner.displacement);
    }
  } else {
    static_assert(is_all_v<Selector>,
                  "layout subview selector must be an index, all, range or "
                  "tuple");
  }
}

template <typename Shape, typename Stride, typename Selectors>
KOKKOS_INLINE_FUNCTION constexpr auto
select_children(const Shape &shape, const Stride &stride,
                const Selectors &selectors, bool enabled) {
  if constexpr (TupleTraits<Selectors>::rank == 0) {
    return make_selected(Tuple<>{}, Tuple<>{}, static_zero{});
  } else {
    const auto head = select_node(tuple_head(shape), tuple_head(stride),
                                  tuple_head(selectors), enabled);
    const auto tail = select_children(tuple_tail(shape), tuple_tail(stride),
                                      tuple_tail(selectors), enabled);
    return make_selected(concatenate(head.shape, tail.shape),
                         concatenate(head.strides, tail.strides),
                         sum(head.displacement, tail.displacement));
  }
}

// Position of parent child J among the children kept in the result.
template <typename Selectors> struct KeptBefore;
template <typename... Selectors> struct KeptBefore<Tuple<Selectors...>> {
  template <std::size_t J>
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t value() {
    if constexpr (J == 0) {
      return 0;
    } else {
      constexpr bool kept[] = { !is_index_v<Selectors>..., false };
      std::size_t count = 0;
      for (std::size_t i = 0; i < J; ++i) {
        count += kept[i] ? 1 : 0;
      }
      return count;
    }
  }
};

template <typename ResultShape, typename Selectors, typename Result,
          std::size_t... J>
KOKKOS_INLINE_FUNCTION constexpr auto
back_children(const ResultShape &result_shape, const Selectors &selectors,
              const Result &result, bool &ok, std::index_sequence<J...>);

template <typename ResultShape, typename Selector, typename Result>
KOKKOS_INLINE_FUNCTION constexpr auto
back_node(const ResultShape &shape, const Selector &selector,
          const Result &result, bool &ok) {
  if constexpr (is_range_v<Selector>) {
    const std::size_t index = linearize(shape, result);
    if constexpr (TupleTraits<ResultShape>::is_tuple) {
      // Leaf ranges need no check: their envelope is exactly the range.
      ok = ok && index < integer(range_count(selector));
    }
    return index + unsigned_value(selector.begin);
  } else if constexpr (TupleTraits<Selector>::is_tuple) {
    return back_children(
        shape, selector, result, ok,
        std::make_index_sequence<TupleTraits<Selector>::rank>{});
  } else {
    return result;
  }
}

template <std::size_t J, typename ResultShape, typename Selectors,
          typename Result>
KOKKOS_INLINE_FUNCTION constexpr auto
back_child(const ResultShape &result_shape, const Selectors &selectors,
           const Result &result, bool &ok) {
  const auto &selector = layout::get<J>(selectors);
  using selector_type = std::remove_cvref_t<decltype(selector)>;
  if constexpr (is_index_v<selector_type>) {
    return integer(selector);
  } else {
    constexpr std::size_t r = KeptBefore<Selectors>::template value<J>();
    return back_node(layout::get<r>(result_shape), selector,
                     layout::get<r>(result), ok);
  }
}

// Map an expanded result coordinate to a parent coordinate. Fixed nodes insert
// their index; ok becomes false if a range restriction rejects the coordinate.
template <typename ResultShape, typename Selectors, typename Result,
          std::size_t... J>
KOKKOS_INLINE_FUNCTION constexpr auto
back_children(const ResultShape &result_shape, const Selectors &selectors,
              const Result &result, bool &ok, std::index_sequence<J...>) {
  return layout::make_tuple(
      back_child<J>(result_shape, selectors, result, ok)...);
}

template <typename Selector>
KOKKOS_INLINE_FUNCTION constexpr bool
ranges_nonempty(const Selector &selector) {
  if constexpr (is_range_v<Selector>) {
    return integer(range_count(selector)) != 0;
  } else if constexpr (TupleTraits<Selector>::is_tuple) {
    if constexpr (TupleTraits<Selector>::rank == 0) {
      return true;
    } else {
      return ranges_nonempty(tuple_head(selector)) &&
             ranges_nonempty(tuple_tail(selector));
    }
  } else {
    return true;
  }
}

template <typename Selectors> struct SubviewTransport {
  Selectors selectors;
  KOKKOS_INLINE_FUNCTION static constexpr const char *message() {
    return "layout subview coordinate out of bounds";
  }
  template <typename Parent, typename Descriptor>
  KOKKOS_INLINE_FUNCTION constexpr auto
  span(const Parent &parent, const Descriptor &descriptor) const {
    return Kokkos::min(descriptor.required_span_size(),
                       parent.required_span_size());
  }
  template <typename Shape, typename ParentShape, typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr auto
  back(const Shape &shape, const ParentShape &, const Coordinate &coord,
       bool &ok) const {
    return back_children(
        shape, selectors, coord, ok,
        std::make_index_sequence<TupleTraits<Selectors>::rank>{});
  }
};

template <typename Selectors> struct Subview {
  Selectors selectors{};

  template <typename Parent>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const Parent &parent) const {
    using shape_type = std::remove_cvref_t<decltype(parent.shape())>;
    constexpr bool matching = selector_matches<shape_type, Selectors>();
    static_assert(matching,
                  "layout subview needs one selector per top-level mode");
    if constexpr (matching) {
      const bool enabled = !parent.empty() && ranges_nonempty(selectors);
      const auto selected =
          select_children(parent.shape(), parent.strides(), selectors, enabled);
      const auto descriptor = LayoutAccess::derive(
          selected.shape, selected.strides, enabled,
          sum(parent.displacement(), selected.displacement));
      return DerivedMapping<Parent, std::remove_cvref_t<decltype(descriptor)>,
                            SubviewTransport<Selectors>>(parent, descriptor,
                                                         { selectors });
    }
  }
};

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Construct a half-open range selector.
 * @tparam Begin Non-boolean integer or integral constant type.
 * @tparam End Non-boolean integer or integral constant type.
 * @param begin First selected scalar index.
 * @param end One past the last selected scalar index.
 * @return Range selector for subview.
 */
template <typename Begin, typename End>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr Range<Begin, End>
range(Begin begin, End end) {
  static_assert(layout_impl::is_index_v<Begin> && layout_impl::is_index_v<End>,
                "layout subview range bounds must be integers");
  if constexpr (layout_impl::static_range_v<Range<Begin, End>>) {
    static_assert(layout_impl::nonnegative(Begin{}) &&
                      layout_impl::nonnegative(End{}) &&
                      layout_impl::ordered(layout_impl::unsigned_value(Begin{}),
                                           layout_impl::unsigned_value(End{})),
                  "layout subview range out of order or out of range");
  }
  return { begin, end };
}

/**
 * @brief Select part of a mapping or view without copying elements.
 *
 * Each selector applies to one node. An integer or integral constant fixes the
 * node's scalar index and removes the node. `all` keeps the node. `range(b, e)`
 * keeps the scalar indices [b, e) and the node's topology. A tuple of
 * selectors, one per child, selects inside a nested node and keeps its level.
 * On a nested node, a nonempty range must start at a multiple of the product
 * of all leaves except the slowest; the end may be unaligned, in which case
 * the envelope rounds up to whole blocks and valid() excludes the tail. Every
 * valid result coordinate corresponds to one valid parent coordinate with the
 * same offset, so parent restrictions carry over.
 *
 * Statically detectable errors fail compilation. Runtime-invalid selectors
 * abort in debug builds and are undefined behavior in release builds.
 * @tparam Object Mapping or layout View type.
 * @tparam Selectors One selector type per top-level mode.
 * @param object Parent whose offsets, restrictions and storage are retained.
 * @param selectors One selector per top-level mode.
 * @return Mapping or view of the selected region, with offsets relative to the
 * parent's original backing pointer. A view result retains the parent's
 * backing handle, capacity and element constness.
 */
template <typename Object, typename... Selectors>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
subview(const Object &object, Selectors... selectors) {
  return layout_impl::transform_object(
      object, layout_impl::Subview<layout_impl::Tuple<Selectors...>>{
                  layout_impl::tuple_make(selectors...) });
}

} // namespace specfem::experimental::layout
