#pragma once

#include "specfem/experimental/layout/kokkos.hpp"

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename T> struct PhysicalRoot : std::false_type {};
template <typename T, typename Mapping, typename Storage, bool Transformed>
struct PhysicalRoot<layout::View<T, Mapping, Storage, Transformed>>
    : std::bool_constant<
          layout::View<T, Mapping, Storage, Transformed>::is_physical_root> {};

template <typename A, typename B>
inline constexpr bool physical_topology_compatible = matching_topology<
    std::remove_cvref_t<decltype(std::declval<A>().layout().shape())>,
    std::remove_cvref_t<decltype(std::declval<B>().layout().shape())>>();

template <typename View>
concept MutableRoot =
    PhysicalRoot<View>::value && !std::is_const_v<typename View::value_type>;

template <typename DstView, typename SrcView>
concept PhysicalCopy = MutableRoot<DstView> && PhysicalRoot<SrcView>::value &&
                       std::is_same_v<typename DstView::non_const_value_type,
                                      typename SrcView::non_const_value_type> &&
                       physical_topology_compatible<DstView, SrcView>;

template <typename A, typename B>
constexpr bool equal_physical_tree(const A &a, const B &b) {
  if constexpr (layout::is_tuple_v<A>) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return (equal_physical_tree(layout::get<I>(a), layout::get<I>(b)) && ...);
    }(std::make_index_sequence<layout::tuple_size_v<A>>{});
  } else {
    return integer(a) == integer(b);
  }
}

template <typename DstView, typename SrcView>
void require_physical_descriptor([[maybe_unused]] const DstView &destination,
                                 [[maybe_unused]] const SrcView &source) {
#ifndef NDEBUG
  const auto &a = destination.layout();
  const auto &b = source.layout();
  bool equal = equal_physical_tree(a.shape(), b.shape()) &&
               equal_physical_tree(a.strides(), b.strides()) &&
               a.empty() == b.empty() &&
               destination.physical_span_size() == source.physical_span_size();
  for (std::size_t i = 0; i < DstView::rank(); ++i) {
    equal = equal && a.extent(i) == b.extent(i);
  }
  require(equal, "layout physical descriptor mismatch");
#endif
}

template <typename View> void require_physical_region(const View &view) {
  require(view.storage_span_size() == view.physical_span_size(),
          "layout physical operation requires exact storage span");
#ifndef NDEBUG
  (void)multiply(view.storage_span_size(), sizeof(typename View::value_type));
#endif
}

template <typename View> auto physical_storage(const View &view) {
  using Native =
      Kokkos::View<typename View::value_type *, typename View::memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  return Native(view.flat_storage().data(), view.physical_span_size());
}

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Create a mutable host mirror without copying source values.
 * @tparam T Element type. @tparam Mapping Root mapping.
 * @tparam Storage Backing storage. @tparam Transformed Transformation state.
 * @param source Root representing exactly its full physical allocation.
 * @return Mutable host view, sharing only matching host element storage.
 *
 * Raw-pointer roots require host storage. Device borrowing requires an
 * unmanaged native adapter tagged with its memory space. Shared borrowed
 * mirrors retain the caller's lifetime responsibility. Other mirrors allocate
 * without copying. Transformed views are rejected. Runtime span checks run only
 * in debug builds.
 */
template <typename T, typename Mapping, typename Storage, bool Transformed>
  requires View<T, Mapping, Storage, Transformed>::is_physical_root
auto create_mirror_view(const View<T, Mapping, Storage, Transformed> &source) {
  layout_impl::require_physical_region(source);
  using Source = View<T, Mapping, Storage, Transformed>;
  using Mirror = typename Source::host_mirror_type;
  if constexpr (std::is_same_v<Mirror, Source>) {
    return source;
  } else {
    return Mirror("layout_mirror", source.layout());
  }
}

/**
 * @brief Fill every physical element, including padding and holes.
 * @tparam T Mutable element type. @tparam Mapping Root mapping.
 * @tparam Storage Backing storage. @tparam Transformed Transformation state.
 * @param destination Root representing exactly its full physical allocation.
 * @param value Value assigned to every physical element.
 *
 * Uses default Kokkos completion semantics. Complete producer work first and
 * retain borrowed storage until all related asynchronous work completes.
 * Invalid runtime spans are undefined behavior in release builds. A root
 * imported from a strided native subview covers that subview's whole span,
 * so this also writes elements the subview skips.
 */
template <typename T, typename Mapping, typename Storage, bool Transformed>
  requires layout_impl::MutableRoot<View<T, Mapping, Storage, Transformed>>
void fill(const View<T, Mapping, Storage, Transformed> &destination,
          const std::remove_const_t<T> &value) {
  layout_impl::require_physical_region(destination);
  if (destination.physical_span_size() != 0) {
    Kokkos::deep_copy(layout_impl::physical_storage(destination), value);
  }
}

/**
 * @brief Copy the entire physical region between matching root descriptors.
 * @tparam DstView Root with mutable elements. @tparam SrcView Root source.
 * @param destination Exact physical region receiving all elements.
 * @param source Exact physical region with the same descriptor and element
 * type.
 *
 * Descriptor equality compares logical extents, shape/stride hierarchies and
 * values, and full physical spans. Runtime mismatches are diagnosed only in
 * debug builds and are undefined behavior in release builds. Complete producer
 * work before copying. Borrowed storage must outlive related asynchronous work.
 * Raw-pointer roots require host storage; use tagged native device adapters for
 * device borrowing. Partial overlaps follow native Kokkos preconditions.
 */
template <typename DstView, typename SrcView>
  requires layout_impl::PhysicalCopy<DstView, SrcView>
void deep_copy(const DstView &destination, const SrcView &source) {
  layout_impl::require_physical_region(destination);
  layout_impl::require_physical_region(source);
  layout_impl::require_physical_descriptor(destination, source);
  if (destination.physical_span_size() != 0) {
    Kokkos::deep_copy(layout_impl::physical_storage(destination),
                      layout_impl::physical_storage(source));
  }
}

} // namespace specfem::experimental::layout
