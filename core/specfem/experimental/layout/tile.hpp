#pragma once

#include "specfem/experimental/layout/subview.hpp"

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename A, typename B>
KOKKOS_INLINE_FUNCTION constexpr auto tile_product(A a, B b, bool enabled) {
  if constexpr (Constant<A> && Constant<B>) {
    return std::integral_constant<std::size_t,
                                  multiply(integer(A{}), integer(B{}))>{};
  } else {
    return enabled ? multiply(integer(a), integer(b)) : std::size_t{ 0 };
  }
}

// Runs before any division by the size.
template <typename Size>
KOKKOS_INLINE_FUNCTION constexpr void check_tile_size(Size size) {
  if constexpr (Constant<Size>) {
    static_assert(Size::value > 0, "layout tile size must be positive");
  }
  if (checks_enabled()) {
    require(nonnegative(size) && unsigned_value(size) != 0,
            "layout tile size must be positive");
  }
}

template <typename Shape, typename Stride, typename Size>
KOKKOS_INLINE_FUNCTION constexpr auto
tile_leaf(const Shape &shape, const Stride &stride, Size size, bool enabled) {
  check_tile_size(size);
  return make_selected(
      layout::make_tuple(size, block_count(shape, size)),
      layout::make_tuple(stride, tile_product(size, stride, enabled)),
      static_zero{});
}

template <typename Tree>
KOKKOS_INLINE_FUNCTION constexpr std::size_t first_leaf_child() {
  if constexpr (LeafCount<tuple_element_t<0, Tree>>::value != 0) {
    return 0;
  } else {
    return 1 + first_leaf_child<decltype(tuple_tail(Tree{}))>();
  }
}

template <typename Tree>
KOKKOS_INLINE_FUNCTION constexpr auto first_leaf(const Tree &tree) {
  if constexpr (!TupleTraits<Tree>::is_tuple) {
    return tree;
  } else {
    return first_leaf(layout::get<first_leaf_child<Tree>()>(tree));
  }
}

template <typename Tree, typename Value>
KOKKOS_INLINE_FUNCTION constexpr auto replace_first_leaf(const Tree &tree,
                                                         Value value) {
  if constexpr (!TupleTraits<Tree>::is_tuple) {
    return value;
  } else {
    constexpr auto k = first_leaf_child<Tree>();
    return replace_child<k>(
        tree, replace_first_leaf(layout::get<k>(tree), value),
        std::make_index_sequence<TupleTraits<Tree>::rank>{});
  }
}

template <typename Leaf, typename Size>
KOKKOS_INLINE_FUNCTION constexpr auto tile_quotient(Leaf leaf, Size size) {
  if constexpr (Constant<Leaf> && Constant<Size>) {
    static_assert(integer(Size{}) != 0, "layout tile size must be positive");
    static_assert(integer(Leaf{}) % integer(Size{}) == 0,
                  "layout tile size must divide the fastest leaf");
    return std::integral_constant<std::size_t,
                                  integer(Leaf{}) / integer(Size{})>{};
  } else {
    if (checks_enabled()) {
      require(integer(leaf) % integer(size) == 0,
              "layout tile size must divide the fastest leaf");
    }
    return integer(leaf) / integer(size);
  }
}

template <typename Shape, typename Strides, typename Selectors>
KOKKOS_INLINE_FUNCTION constexpr auto
tile_children(const Shape &shape, const Strides &strides,
              const Selectors &selectors, bool enabled);

template <typename Shape, typename Stride, typename Selector>
KOKKOS_INLINE_FUNCTION constexpr auto
tile_node(const Shape &shape, const Stride &stride, const Selector &selector,
          bool enabled) {
  if constexpr (is_all_v<Selector>) {
    return make_selected(shape, stride, static_zero{});
  } else if constexpr (!TupleTraits<Shape>::is_tuple && is_index_v<Selector>) {
    return tile_leaf(shape, stride, selector, enabled);
  } else if constexpr (TupleTraits<Selector>::is_tuple) {
    static_assert(selector_matches<Shape, Selector>(),
                  "layout tile tuple selector must match nested children");
    if constexpr (selector_matches<Shape, Selector>()) {
      return tile_children(shape, stride, selector, enabled);
    }
  } else {
    static_assert(is_index_v<Selector>,
                  "layout tile selector must be a size, all or tuple");
    static_assert(LeafCount<Shape>::value != 0, "layout tile requires a leaf");
    if constexpr (is_index_v<Selector> && LeafCount<Shape>::value != 0) {
      check_tile_size(selector);
      const auto rest = tile_quotient(first_leaf(shape), selector);
      return make_selected(
          layout::make_tuple(selector, replace_first_leaf(shape, rest)),
          layout::make_tuple(
              first_leaf(stride),
              replace_first_leaf(
                  stride, tile_product(selector, first_leaf(stride), enabled))),
          static_zero{});
    }
  }
}

template <typename Shape, typename Strides, typename Selectors>
KOKKOS_INLINE_FUNCTION constexpr auto
tile_children(const Shape &shape, const Strides &strides,
              const Selectors &selectors, bool enabled) {
  if constexpr (TupleTraits<Shape>::rank == 0) {
    return make_selected(layout::make_tuple(), layout::make_tuple(),
                         static_zero{});
  } else {
    const auto head = tile_node(tuple_head(shape), tuple_head(strides),
                                tuple_head(selectors), enabled);
    const auto tail = tile_children(tuple_tail(shape), tuple_tail(strides),
                                    tuple_tail(selectors), enabled);
    return make_selected(tuple_prepend(head.shape, tail.shape),
                         tuple_prepend(head.strides, tail.strides),
                         static_zero{});
  }
}

template <typename Shape, typename ParentShape, typename Selector,
          typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr auto
tile_back(const Shape &shape, const ParentShape &parent,
          const Selector &selector, const Coordinate &coord, bool &ok) {
  if constexpr (is_index_v<Selector>) {
    const auto scalar = linearize(shape, coord);
    if constexpr (!TupleTraits<ParentShape>::is_tuple) {
      // Only a leaf has a rounded tail. A nested node's envelope is exact.
      ok = ok && scalar < integer(parent);
    }
    return scalar;
  } else if constexpr (is_all_v<Selector>) {
    return coord;
  } else if constexpr (TupleTraits<Selector>::rank == 0) {
    return layout::make_tuple();
  } else {
    const auto head = tile_back(tuple_head(shape), tuple_head(parent),
                                tuple_head(selector), tuple_head(coord), ok);
    const auto tail = tile_back(tuple_tail(shape), tuple_tail(parent),
                                tuple_tail(selector), tuple_tail(coord), ok);
    return tuple_prepend(head, tail);
  }
}

template <typename Selectors> struct TileTransport {
  Selectors selectors;
  KOKKOS_INLINE_FUNCTION static constexpr const char *message() {
    return "layout tile coordinate out of bounds";
  }
  template <typename Parent, typename Descriptor>
  KOKKOS_INLINE_FUNCTION constexpr auto span(const Parent &parent,
                                             const Descriptor &) const {
    return parent.required_span_size();
  }
  template <typename Shape, typename ParentShape, typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr auto
  back(const Shape &shape, const ParentShape &parent, const Coordinate &coord,
       bool &ok) const {
    return tile_back(shape, parent, selectors, coord, ok);
  }
};

template <typename Selectors> struct Tiling {
  Selectors selectors;
  template <typename Parent>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const Parent &parent) const {
    using shape_type = std::remove_cvref_t<decltype(parent.shape())>;
    constexpr bool matching = selector_matches<shape_type, Selectors>();
    static_assert(matching,
                  "layout tile needs one selector per top-level mode");
    if constexpr (matching) {
      const auto selected = tile_children(parent.shape(), parent.strides(),
                                          selectors, !parent.empty());
      const auto descriptor =
          LayoutAccess::derive(selected.shape, selected.strides,
                               !parent.empty(), parent.displacement());
      return DerivedMapping<Parent, std::remove_cvref_t<decltype(descriptor)>,
                            TileTransport<Selectors>>(parent, descriptor,
                                                      { selectors });
    }
  }
};
} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {
/**
 * @brief Partition selected modes into nested tiles without copying elements.
 * @tparam Object Mapping or layout View type.
 * @tparam Selectors One size, all, or nested selector tuple per top-level mode.
 * @param object Parent whose validity and backing lifetime are retained.
 * @param selectors Positive sizes, all, or tuples selecting nested children.
 * @return Mapping or view with each tiled node replaced by (size, rest).
 *
 * Check valid() before access. Boundary padding is outside the logical domain.
 * A nested node requires a size dividing its fastest leaf. Static invalid
 * requests fail compilation. Runtime-invalid requests abort in debug builds
 * and have undefined behavior in release builds.
 */
template <typename Object, typename... Selectors>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
tile(const Object &object, Selectors... selectors) {
  return layout_impl::transform_object(
      object,
      layout_impl::Tiling<decltype(layout_impl::tuple_make(selectors...))>{
          layout_impl::tuple_make(selectors...) });
}
} // namespace specfem::experimental::layout
