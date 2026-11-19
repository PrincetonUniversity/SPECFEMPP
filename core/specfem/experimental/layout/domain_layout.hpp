#pragma once

#include "specfem/experimental/layout/layout.hpp"
#include <span>
#include <utility>

/** @cond */
namespace specfem::experimental::layout_impl {
template <typename A, typename B>
KOKKOS_INLINE_FUNCTION constexpr auto domain_product(A a, B b, bool empty) {
  if constexpr (StaticTree<A>::value && StaticTree<B>::value) {
    return layout::c<multiply(product(A{}), product(B{}))>;
  } else {
    return empty ? std::size_t{ 0 } : multiply(product(a), product(b));
  }
}

template <typename Tuple, typename Stride>
KOKKOS_INLINE_FUNCTION constexpr auto
domain_strides(const Tuple &extents, Stride stride, bool empty) {
  if constexpr (TupleTraits<Tuple>::rank == 0) {
    return layout::make_tuple();
  } else {
    return tuple_prepend(
        stride, domain_strides(
                    tuple_tail(extents),
                    domain_product(stride, tuple_head(extents), empty), empty));
  }
}

template <std::size_t B, typename N>
KOKKOS_INLINE_FUNCTION constexpr auto domain_tiles(N n) {
  if constexpr (Constant<N>) {
    return layout::c<integer(N{}) / B + (integer(N{}) % B != 0)>;
  } else {
    const auto count = integer(n);
    return count / B + (count % B != 0);
  }
}

template <std::size_t I, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr auto domain_extent(const Extents &extents) {
  if constexpr (Extents::static_extent(I) != std::dynamic_extent) {
    return layout::c<Extents::static_extent(I)>;
  } else {
    return std::size_t(extents.extent(I));
  }
}

template <typename Extents, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr auto
domain_extents(const Extents &extents, std::index_sequence<I...>) {
  return layout::make_tuple(domain_extent<I>(extents)...);
}
} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {
#if defined(KOKKOS_ENABLE_CUDA)
inline constexpr std::size_t domain_tile_size = 32;
#elif defined(KOKKOS_ENABLE_HIP)
inline constexpr std::size_t domain_tile_size = 64;
#else
inline constexpr std::size_t domain_tile_size = 1;
#endif

/**
 * @brief Construct Left storage, optionally tiling the first logical axis.
 * @tparam B Positive storage tile size, defaulting to the backend tile size.
 * Explicit one selects ordinary LayoutLeft.
 * @tparam Extents Tuple of logical scalar extents or an mdspan extents type.
 * @param extents Logical extents, with static values retained.
 * @return Shared Layout with physical envelope and logical bounds.
 */
template <std::size_t B = domain_tile_size, typename Extents>
KOKKOS_INLINE_FUNCTION constexpr auto
make_domain_layout(const Extents &extents) {
  static_assert(B > 0, "layout storage tile size must be positive");
  if constexpr (!is_tuple_v<Extents>) {
    return make_domain_layout<B>(layout_impl::domain_extents(
        extents, std::make_index_sequence<Extents::rank()>{}));
  } else {
    static_assert(tuple_size_v<Extents> > 0,
                  "domain layout requires positive rank");
    layout_impl::validate_leaves(extents);
    const bool empty = layout_impl::has_zero(extents);
    if constexpr (B == 1) {
      return make_layout(extents, layout_impl::domain_strides(
                                      extents, c<std::size_t{ 1 }>, empty));
    } else {
      const auto tiles =
          layout_impl::domain_tiles<B>(layout_impl::tuple_head(extents));
      const auto tail_strides = layout_impl::domain_strides(
          layout_impl::tuple_tail(extents), c<B>, empty);
      const auto tile_stride = layout_impl::domain_product(
          c<B>, layout_impl::tuple_tail(extents), empty);
      return make_layout(
          layout_impl::tuple_prepend(layout::make_tuple(c<B>, tiles),
                                     layout_impl::tuple_tail(extents)),
          layout_impl::tuple_prepend(
              layout::make_tuple(c<std::size_t{ 1 }>, tile_stride),
              tail_strides),
          extents);
    }
  }
}

/** @brief Backend domain mapping expressed by the shared Layout.
 * @tparam Extents Logical extents type.
 * @tparam B Storage tile size, independent of execution chunks. */
template <typename Extents, std::size_t B = domain_tile_size>
using DomainLayout = decltype(make_domain_layout<B>(std::declval<Extents>()));
} // namespace specfem::experimental::layout

/** @cond */
namespace specfem::experimental::layout_impl {
template <typename Target, typename Value>
KOKKOS_INLINE_FUNCTION constexpr auto domain_value(Value value) {
  if constexpr (Constant<Target>) {
    if constexpr (Constant<Value>) {
      static_assert(integer(Target{}) == integer(Value{}),
                    "layout static extent mismatch");
    }
    require(integer(value) == integer(Target{}),
            "layout static extent mismatch");
    return Target{};
  } else {
    if constexpr (Constant<Value>) {
      static_assert(integer(Value{}) <= std::numeric_limits<Target>::max(),
                    "layout logical extent exceeds leaf range");
    }
    require(integer(value) <= std::numeric_limits<Target>::max(),
            "layout logical extent exceeds leaf range");
    return static_cast<Target>(integer(value));
  }
}

template <typename Mapping, typename Indices, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr auto
make_domain_layout_like_impl(const Indices &indices,
                             std::index_sequence<I...>) {
  using bounds =
      std::conditional_t<std::is_void_v<typename Mapping::logical_bounds_type>,
                         typename Mapping::shape_type,
                         typename Mapping::logical_bounds_type>;
  const auto logical = layout::make_tuple(
      domain_value<std::remove_cvref_t<decltype(layout::get<I>(bounds{}))>>(
          layout::get<I>(indices))...);
  using shape = typename Mapping::shape_type;
  using first = std::remove_cvref_t<decltype(layout::get<0>(shape{}))>;
  constexpr std::size_t tile_size = [] {
    if constexpr (TupleTraits<first>::is_tuple) {
      return integer(std::remove_cvref_t<decltype(layout::get<0>(first{}))>{});
    } else {
      return std::size_t{ 1 };
    }
  }();
  const auto result = layout::make_domain_layout<tile_size>(logical);
  static_assert(std::is_same_v<Mapping, std::remove_cvref_t<decltype(result)>>,
                "layout extents constructor requires canonical domain layout");
  return result;
}

template <typename Mapping, typename... Indices>
KOKKOS_INLINE_FUNCTION constexpr auto
make_domain_layout_like(Indices... indices) {
  static_assert(sizeof...(Indices) == Mapping::rank(),
                "layout coordinate rank mismatch");
  return make_domain_layout_like_impl<Mapping>(
      layout::make_tuple(indices...),
      std::make_index_sequence<sizeof...(Indices)>{});
}
} // namespace specfem::experimental::layout_impl
/** @endcond */
