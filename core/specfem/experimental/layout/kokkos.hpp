#pragma once

#include "specfem/experimental/layout/view.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename NativeView> struct KokkosStorage {
  using element_type = std::remove_pointer_t<typename NativeView::pointer_type>;
  static_assert(
      std::is_same_v<typename NativeView::reference_type, element_type &>,
      "layout native adapter requires ordinary element references");
  static_assert(
      std::is_same_v<typename NativeView::array_layout, Kokkos::LayoutLeft> ||
          std::is_same_v<typename NativeView::array_layout,
                         Kokkos::LayoutRight> ||
          std::is_same_v<typename NativeView::array_layout,
                         Kokkos::LayoutStride>,
      "layout native adapter requires LayoutLeft, LayoutRight or LayoutStride");
  NativeView native;

  KOKKOS_INLINE_FUNCTION auto data() const { return native.data(); }
  KOKKOS_INLINE_FUNCTION std::size_t size() const { return native.span(); }
  KOKKOS_INLINE_FUNCTION typename NativeView::reference_type
  access(std::size_t offset) const {
#ifndef NDEBUG
    using memory_space = typename NativeView::memory_space;
    KOKKOS_IF_ON_HOST(
        (require(Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                            memory_space>::accessible,
                 "layout access to inaccessible memory space");))
    KOKKOS_IF_ON_DEVICE(
        (require(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                            memory_space>::accessible,
                 "layout access to inaccessible memory space");))
#endif
    return native.data()[offset];
  }
};

template <std::size_t I, typename NativeView>
KOKKOS_INLINE_FUNCTION auto native_extent(const NativeView &native) {
  if constexpr (NativeView::static_extent(I) != 0) {
    return layout::c<NativeView::static_extent(I)>;
  } else {
    return native.extent(I);
  }
}

template <typename NativeView, std::size_t... I>
KOKKOS_INLINE_FUNCTION auto native_layout(const NativeView &native,
                                          std::index_sequence<I...>) {
  return layout::make_layout(layout::make_tuple(native_extent<I>(native)...),
                             layout::make_tuple(native.stride(I)...));
}

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Interpret native storage through a compatible certified mapping.
 * @tparam DataType Native element and rank description.
 * @tparam Properties Native layout, memory space and traits.
 * @tparam Mapping Certified mapping relative to native.data().
 * @param native Original handle, retained without changing its ownership.
 * @param mapping Mapping whose required span fits native.span().
 * @return View retaining native constness, memory space and backing lifetime.
 *
 * Unmanaged storage remains borrowed. The caller must retain backing storage
 * until asynchronous work completes. No allocation or element copy occurs.
 */
template <typename DataType, typename... Properties, typename Mapping>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
make_view(const Kokkos::View<DataType, Properties...> &native,
          const Mapping &mapping) {
  using native_type = Kokkos::View<DataType, Properties...>;
  using element_type =
      std::remove_pointer_t<typename native_type::pointer_type>;
  return layout_impl::ViewAccess::make_view<element_type>(
      layout_impl::KokkosStorage<native_type>{ native }, mapping);
}

/**
 * @brief Import native extents and actual strides without copying elements.
 * @tparam DataType Native element and rank description.
 * @tparam Properties Native layout, memory space and traits.
 * @param native Original managed or borrowed native handle.
 * @return Certified view with the native coordinate domain and ownership.
 *
 * Static extents retain their type; strides come from the native handle.
 * Subview offsets are relative to the subview's own data pointer.
 */
template <typename DataType, typename... Properties>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto
make_view(const Kokkos::View<DataType, Properties...> &native) {
  using native_type = Kokkos::View<DataType, Properties...>;
  return make_view(
      native, layout_impl::native_layout(
                  native, std::make_index_sequence<native_type::rank()>{}));
}

} // namespace specfem::experimental::layout
