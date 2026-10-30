#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::datatype {
template <typename T, typename Extents, typename Layout, typename MemorySpace>
class View;
} // namespace specfem::datatype

namespace specfem::io_backends::impl::NPY {

template <typename ViewType> struct is_domain_view : std::false_type {};

template <typename T, typename Extents, typename Layout, typename MemorySpace>
struct is_domain_view<specfem::datatype::View<T, Extents, Layout, MemorySpace>>
    : std::true_type {};

/**
 * @brief Whether the buffer of @p ViewType is described as Fortran
 * (column-major) order in an NPY header
 *
 * The buffer is written as is, so the header must describe the view's own
 * layout: column-major for `Kokkos::LayoutLeft`, row-major for
 * `Kokkos::LayoutRight`. Rank 0 and 1 views have no order.
 *
 * Tiled domain views (`specfem::datatype::View`) match neither layout and keep
 * the column-major header; their inherited `array_layout` does not describe
 * the tiling.
 *
 * @tparam ViewType View passed to an NPY or NPZ dataset
 * @return true if the header must declare `fortran_order: True`
 */
template <typename ViewType> constexpr bool is_fortran_order() {
  if constexpr (is_domain_view<ViewType>::value) {
    return true;
  } else if constexpr (ViewType::rank() <= 1) {
    return true;
  } else {
    using layout = typename ViewType::array_layout;
    static_assert(std::is_same_v<layout, Kokkos::LayoutLeft> ||
                      std::is_same_v<layout, Kokkos::LayoutRight>,
                  "NPY/NPZ can only describe contiguous LayoutLeft or "
                  "LayoutRight views");
    return std::is_same_v<layout, Kokkos::LayoutLeft>;
  }
}

} // namespace specfem::io_backends::impl::NPY
