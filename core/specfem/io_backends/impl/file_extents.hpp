#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace specfem {
namespace io_backends {
namespace impl {

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
