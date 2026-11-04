#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace specfem::datatype {
template <typename T, typename Extents, typename Layout, typename MemorySpace>
class View;
} // namespace specfem::datatype

namespace specfem {
namespace io_backends {
namespace impl {

/**
 * @brief Whether @p ViewType is a tiled `specfem::datatype::View`
 * (DomainView), including its host mirrors.
 *
 * @tparam ViewType View type to test
 */
template <typename ViewType> struct is_domain_view : std::false_type {};

template <typename T, typename Extents, typename Layout, typename MemorySpace>
struct is_domain_view<specfem::datatype::View<T, Extents, Layout, MemorySpace>>
    : std::true_type {};

/**
 * @brief Compile-time guard against handing a tiled DomainView to a dataset.
 *
 * The raw buffer of a DomainView is tiled and must not be written as is.
 *
 * @tparam ViewType View type passed to a dataset
 */
template <typename ViewType> constexpr void assert_not_domain_view() {
  static_assert(!is_domain_view<ViewType>::value,
                "specfem::datatype::View (DomainView) storage is tiled; pack "
                "it with specfem::io::impl::extract_sub_block or write "
                "get_base_view() instead of passing it to a dataset");
}

/**
 * @brief Extents of @p view as they must be declared in a row-major file
 * format.
 *
 * LayoutLeft views of rank >= 2 are reported with reversed extents because
 * their contiguous buffer is exactly the row-major array of the reversed
 * shape. Everything else (rank 0/1, LayoutRight, ...) is reported as is.
 *
 * @tparam ViewType Kokkos view type
 * @param view View whose extents are queried
 * @return Extents to declare in the file
 */
template <typename ViewType>
std::vector<std::size_t> file_extents(const ViewType &view) {
  constexpr int rank = ViewType::rank();
  std::vector<std::size_t> extents(rank);
  for (int i = 0; i < rank; ++i) {
    extents[i] = view.extent(i);
  }
  if constexpr (std::is_same_v<typename ViewType::array_layout,
                               Kokkos::LayoutLeft> &&
                (rank >= 2)) {
    std::reverse(extents.begin(), extents.end());
  }
  return extents;
}

} // namespace impl
} // namespace io_backends
} // namespace specfem
