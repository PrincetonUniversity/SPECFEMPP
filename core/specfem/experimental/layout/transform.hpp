#pragma once

#include "specfem/experimental/layout/view.hpp"
#include <utility>

namespace specfem::experimental::layout {

/**
 * @brief Compile-time child indices selecting a node in a layout hierarchy.
 * @tparam Indices Child positions. An empty path selects the root tuple.
 */
template <std::size_t... Indices> struct ModePath {};

} // namespace specfem::experimental::layout

/** @cond */
namespace specfem::experimental::layout_impl {

template <std::size_t I, typename Tree, typename Child, std::size_t... J>
KOKKOS_INLINE_FUNCTION constexpr auto
replace_child(const Tree &tree, const Child &child, std::index_sequence<J...>);

template <std::size_t I, std::size_t J, typename Tree, typename Child>
KOKKOS_INLINE_FUNCTION constexpr auto select_child(const Tree &tree,
                                                   const Child &child) {
  if constexpr (I == J) {
    return child;
  } else {
    return layout::get<J>(tree);
  }
}

template <std::size_t I, typename Tree, typename Child, std::size_t... J>
KOKKOS_INLINE_FUNCTION constexpr auto
replace_child(const Tree &tree, const Child &child, std::index_sequence<J...>) {
  return layout::make_tuple(select_child<I, J>(tree, child)...);
}

template <typename Path, typename Operation> struct AtPath;

template <typename Operation> struct AtPath<layout::ModePath<>, Operation> {
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &tree) {
    return Operation::forward(tree);
  }
  template <typename Shape, typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto backward(const Shape &shape,
                                                        const Tree &tree) {
    return Operation::backward(shape, tree);
  }
};

template <std::size_t I, std::size_t... Rest, typename Operation>
struct AtPath<layout::ModePath<I, Rest...>, Operation> {
  using next = AtPath<layout::ModePath<Rest...>, Operation>;
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &tree) {
    static_assert(TupleTraits<Tree>::is_tuple,
                  "layout transform path must select tuple children");
    if constexpr (TupleTraits<Tree>::is_tuple) {
      static_assert(I < TupleTraits<Tree>::rank,
                    "layout transform path out of range");
      if constexpr (I < TupleTraits<Tree>::rank) {
        return replace_child<I>(
            tree, next::forward(layout::get<I>(tree)),
            std::make_index_sequence<TupleTraits<Tree>::rank>{});
      }
    }
  }
  template <typename Shape, typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto backward(const Shape &shape,
                                                        const Tree &tree) {
    return replace_child<I>(
        tree, next::backward(layout::get<I>(shape), layout::get<I>(tree)),
        std::make_index_sequence<TupleTraits<Tree>::rank>{});
  }
};

template <std::size_t... Order> struct Permutation {
  template <std::size_t Rank>
  KOKKOS_INLINE_FUNCTION static constexpr bool correct_order() {
    if constexpr (sizeof...(Order) != Rank) {
      return false;
    } else if constexpr (Rank == 0) {
      return true;
    } else {
      bool seen[Rank == 0 ? 1 : Rank]{};
      const std::size_t order[] = { Order..., 0 };
      for (std::size_t i = 0; i < Rank; ++i) {
        if (order[i] >= Rank || seen[order[i]]) {
          return false;
        }
        seen[order[i]] = true;
      }
      return true;
    }
  }
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &tree) {
    static_assert(TupleTraits<Tree>::is_tuple,
                  "layout permutation requires a tuple");
    if constexpr (TupleTraits<Tree>::is_tuple) {
      constexpr bool valid = correct_order<TupleTraits<Tree>::rank>();
      static_assert(valid, "layout permutation must list every child once");
      if constexpr (valid) {
        return layout::make_tuple(layout::get<Order>(tree)...);
      }
    }
  }
  template <std::size_t I>
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t position() {
    constexpr std::size_t order[] = { Order..., 0 };
    for (std::size_t j = 0; j < sizeof...(Order); ++j) {
      if (order[j] == I) {
        return j;
      }
    }
    return 0;
  }
  template <typename Tree, std::size_t... I>
  KOKKOS_INLINE_FUNCTION static constexpr auto
  inverse(const Tree &tree, std::index_sequence<I...>) {
    return layout::make_tuple(layout::get<position<I>()>(tree)...);
  }
  template <typename Shape, typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto backward(const Shape &,
                                                        const Tree &tree) {
    return inverse(tree, std::make_index_sequence<sizeof...(Order)>{});
  }
};

template <typename... A, typename... B>
KOKKOS_INLINE_FUNCTION constexpr auto concatenate(const Tuple<A...> &a,
                                                  const Tuple<B...> &b) {
  if constexpr (sizeof...(A) == 0) {
    return b;
  } else {
    return tuple_prepend(tuple_head(a), concatenate(tuple_tail(a), b));
  }
}

template <std::size_t Begin, typename Tree, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr auto children(const Tree &tree,
                                               std::index_sequence<I...>) {
  return layout::make_tuple(layout::get<Begin + I>(tree)...);
}

template <std::size_t First, std::size_t Count> struct Grouping {
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &tree) {
    static_assert(TupleTraits<Tree>::is_tuple, "layout group requires a tuple");
    if constexpr (TupleTraits<Tree>::is_tuple) {
      constexpr auto rank = TupleTraits<Tree>::rank;
      static_assert(First <= rank && Count > 0 && Count <= rank - First,
                    "layout group interval out of range or empty");
      if constexpr (First <= rank && Count > 0 && Count <= rank - First) {
        return concatenate(
            children<0>(tree, std::make_index_sequence<First>{}),
            tuple_prepend(
                children<First>(tree, std::make_index_sequence<Count>{}),
                children<First + Count>(
                    tree, std::make_index_sequence<rank - First - Count>{})));
      }
    }
  }
  template <typename Shape, typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto backward(const Shape &,
                                                        const Tree &tree) {
    return concatenate(
        children<0>(tree, std::make_index_sequence<First>{}),
        concatenate(layout::get<First>(tree),
                    children<First + 1>(
                        tree, std::make_index_sequence<TupleTraits<Tree>::rank -
                                                       First - 1>{})));
  }
};

template <std::size_t I> struct Ungrouping {
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &tree) {
    static_assert(TupleTraits<Tree>::is_tuple,
                  "layout ungroup requires a tuple parent");
    if constexpr (TupleTraits<Tree>::is_tuple) {
      static_assert(I < TupleTraits<Tree>::rank,
                    "layout ungroup path out of range");
      if constexpr (I < TupleTraits<Tree>::rank) {
        using child = std::remove_cvref_t<decltype(layout::get<I>(tree))>;
        static_assert(TupleTraits<child>::is_tuple,
                      "layout ungroup requires a tuple child");
        if constexpr (TupleTraits<child>::is_tuple) {
          return concatenate(
              children<0>(tree, std::make_index_sequence<I>{}),
              concatenate(layout::get<I>(tree),
                          children<I + 1>(
                              tree, std::make_index_sequence<
                                        TupleTraits<Tree>::rank - I - 1>{})));
        }
      }
    }
  }
  template <typename Shape, typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto backward(const Shape &shape,
                                                        const Tree &tree) {
    using child = std::remove_cvref_t<decltype(layout::get<I>(shape))>;
    constexpr auto count = TupleTraits<child>::rank;
    // Unlike public grouping, inverse ungrouping can insert an empty tuple.
    return concatenate(
        children<0>(tree, std::make_index_sequence<I>{}),
        tuple_prepend(children<I>(tree, std::make_index_sequence<count>{}),
                      children<I + count>(
                          tree, std::make_index_sequence<
                                    TupleTraits<Tree>::rank - I - count>{})));
  }
};

template <typename Path> struct UngroupOperation;
template <> struct UngroupOperation<layout::ModePath<>> {
  template <typename Tree>
  KOKKOS_INLINE_FUNCTION static constexpr auto forward(const Tree &) {
    static_assert(!std::is_same_v<Tree, Tree>,
                  "layout ungroup cannot select root");
  }
};
template <std::size_t I>
struct UngroupOperation<layout::ModePath<I>> : Ungrouping<I> {};
template <std::size_t I, std::size_t J, std::size_t... Rest>
struct UngroupOperation<layout::ModePath<I, J, Rest...>>
    : AtPath<layout::ModePath<I>,
             UngroupOperation<layout::ModePath<J, Rest...>>> {};

// Expand scalar subtrees only after their complete envelope bounds pass.
template <typename Shape, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr auto
expand_coordinate(const Shape &shape, const Coordinate &coord);

template <typename Shape>
KOKKOS_INLINE_FUNCTION constexpr auto expand_scalar(const Shape &shape,
                                                    std::size_t index) {
  if constexpr (!TupleTraits<Shape>::is_tuple) {
    return index;
  } else if constexpr (TupleTraits<Shape>::rank == 0) {
    return layout::make_tuple();
  } else {
    const auto extent = product(tuple_head(shape));
    return tuple_prepend(expand_scalar(tuple_head(shape), index % extent),
                         expand_scalar(tuple_tail(shape), index / extent));
  }
}

template <typename Shape, typename Coordinate>
KOKKOS_INLINE_FUNCTION constexpr auto
expand_coordinate(const Shape &shape, const Coordinate &coord) {
  if constexpr (!TupleTraits<Coordinate>::is_tuple) {
    return expand_scalar(shape, integer(coord));
  } else if constexpr (TupleTraits<Coordinate>::rank == 0) {
    return layout::make_tuple();
  } else {
    return tuple_prepend(
        expand_coordinate(tuple_head(shape), tuple_head(coord)),
        expand_coordinate(tuple_tail(shape), tuple_tail(coord)));
  }
}

template <typename Parent, typename Descriptor, typename Transport>
class DerivedMapping {
public:
  static constexpr bool is_transformed = true;
  KOKKOS_INLINE_FUNCTION constexpr std::size_t physical_span_size() const {
    if constexpr (requires { parent_.physical_span_size(); }) {
      return parent_.physical_span_size();
    } else {
      return parent_.required_span_size();
    }
  }
  KOKKOS_INLINE_FUNCTION constexpr DerivedMapping(Parent parent,
                                                  Descriptor descriptor,
                                                  Transport transport = {})
      : parent_(parent), descriptor_(descriptor), transport_(transport) {}
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t rank() {
    return Descriptor::rank();
  }
  KOKKOS_INLINE_FUNCTION constexpr const auto &shape() const {
    return descriptor_.shape();
  }
  KOKKOS_INLINE_FUNCTION constexpr const auto &strides() const {
    return descriptor_.strides();
  }
  KOKKOS_INLINE_FUNCTION constexpr auto displacement() const {
    return descriptor_.displacement();
  }
  template <std::size_t I>
  KOKKOS_INLINE_FUNCTION constexpr auto extent() const {
    return descriptor_.template extent<I>();
  }
  KOKKOS_INLINE_FUNCTION constexpr auto envelope_size() const {
    return descriptor_.envelope_size();
  }
  KOKKOS_INLINE_FUNCTION constexpr bool empty() const {
    return parent_.empty() || descriptor_.empty();
  }
  KOKKOS_INLINE_FUNCTION constexpr std::size_t required_span_size() const {
    return transport_.span(parent_, descriptor_);
  }
  template <typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr bool valid(const Coordinate &coord) const {
    if (!descriptor_.valid(coord)) {
      return false;
    }
    bool ok = true;
    const auto back = transport_.back(shape(), parent_.shape(),
                                      expand_coordinate(shape(), coord), ok);
    return ok && parent_.valid(back);
  }
  template <typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr std::size_t
  offset(const Coordinate &coord) const {
    if (checks_enabled()) {
      require(valid(coord), Transport::message());
    }
    return descriptor_.offset(coord);
  }

private:
  Parent parent_;
  Descriptor descriptor_;
  Transport transport_;
};

template <typename Operation> struct ReorganizationTransport {
  KOKKOS_INLINE_FUNCTION static constexpr const char *message() {
    return "layout transformed coordinate out of bounds";
  }
  template <typename Parent, typename Descriptor>
  KOKKOS_INLINE_FUNCTION constexpr auto span(const Parent &parent,
                                             const Descriptor &) const {
    return parent.required_span_size();
  }
  template <typename Shape, typename ParentShape, typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr auto
  back(const Shape &, const ParentShape &parent, const Coordinate &coord,
       bool &) const {
    return Operation::backward(parent, coord);
  }
};

// Apply a mapping transformation to a mapping, or to a view's mapping while
// retaining the view's backing storage through the single rewrap operation.
template <typename Object, typename Transformation>
KOKKOS_INLINE_FUNCTION constexpr auto
transform_object(const Object &object, const Transformation &transformation) {
  if constexpr (requires { object.layout(); }) {
    return ViewAccess::rebind_layout(object, transformation(object.layout()));
  } else {
    static_assert(Mapping<Object>,
                  "layout transformation requires a mapping or layout view");
    return transformation(object);
  }
}

template <typename Operation> struct Reorganization {
  template <typename Parent>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const Parent &parent) const {
    const auto descriptor =
        LayoutAccess::derive(Operation::forward(parent.shape()),
                             Operation::forward(parent.strides()),
                             !parent.empty(), parent.displacement());
    return DerivedMapping<Parent, std::remove_cvref_t<decltype(descriptor)>,
                          ReorganizationTransport<Operation>>(parent,
                                                              descriptor);
  }
};

template <typename Operation, typename Object>
KOKKOS_INLINE_FUNCTION constexpr auto reorganize(const Object &object) {
  return transform_object(object, Reorganization<Operation>{});
}

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Reorder all immediate children of a selected tuple without copying.
 * @tparam Path ModePath selecting a tuple, including the root.
 * @tparam Order Every child index exactly once in its new order.
 * @tparam Object Mapping or layout View type.
 * @param object Parent whose domain, offsets and storage lifetime are retained.
 * @return Mapping or view with reordered metadata and transported validity.
 *
 * Scalar coordinates use first-leaf-fastest order in the resulting hierarchy.
 * Invalid static selections fail compilation. Runtime-invalid access fails in
 * debug builds and is undefined behavior in release builds.
 */
template <typename Path, std::size_t... Order, typename Object>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
permute(const Object &object) {
  return layout_impl::reorganize<
      layout_impl::AtPath<Path, layout_impl::Permutation<Order...>>>(object);
}

/**
 * @brief Group consecutive children into one nested mode without copying.
 * @tparam Path ModePath selecting the tuple containing the children.
 * @tparam First First child to group.
 * @tparam Count Positive number of children to group. One adds a tuple level.
 * @tparam Object Mapping or layout View type.
 * @param object Parent whose offsets, domain and ownership are retained.
 * @return Mapping or view with a nested mode in first-leaf-fastest order.
 *
 * Invalid static intervals fail compilation. Nonconsecutive grouping requires
 * permutation first. Runtime-invalid access has the same rules as permute.
 */
template <typename Path, std::size_t First, std::size_t Count, typename Object>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
group(const Object &object) {
  return layout_impl::reorganize<
      layout_impl::AtPath<Path, layout_impl::Grouping<First, Count>>>(object);
}

/**
 * @brief Replace a nonroot tuple mode with its immediate children, without
 * copying.
 * @tparam Path ModePath selecting a tuple child. Root and leaf paths are
 * invalid.
 * @tparam Object Mapping or layout View type.
 * @param object Parent whose offsets, domain and ownership are retained.
 * @return Mapping or view with the selected tuple level removed.
 *
 * Deeper descendants retain their nesting. An empty tuple child is removed.
 * Runtime-invalid access has the same rules as permute.
 */
template <typename Path, typename Object>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
ungroup(const Object &object) {
  return layout_impl::reorganize<layout_impl::UngroupOperation<Path>>(object);
}

} // namespace specfem::experimental::layout
