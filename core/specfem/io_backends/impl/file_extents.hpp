#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace io_backends {
namespace impl {

/**
 * @brief Order in which a contiguous view buffer stores its elements.
 */
enum class StorageOrder {
  row_major,   ///< Last index varies fastest (LayoutRight strides)
  column_major ///< First index varies fastest (LayoutLeft strides)
};

/**
 * @brief Classify a view whose buffer can be written as is.
 *
 * The buffer is writable when it is contiguous (span equals the product of
 * the extents) and its strides are the LayoutRight or LayoutLeft strides of
 * its extents. Rank 0 and 1 views are row-major. When both stride sets match
 * (every extent is 1), row-major is reported, as NumPy does.
 *
 * @tparam ViewType View type with `rank()`, `extent()`, `stride()`, `span()`
 * @param view View to classify
 * @return Storage order of the buffer, or `std::nullopt` if it cannot be
 * written as is
 */
template <typename ViewType>
std::optional<StorageOrder> contiguous_storage_order(const ViewType &view) {
  constexpr std::size_t rank = ViewType::rank();
  std::array<std::size_t, rank> extents{};
  std::size_t size = 1;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    extents[axis] = view.extent(axis);
    size *= extents[axis];
  }
  if (static_cast<std::size_t>(view.span()) != size) {
    return std::nullopt;
  }
  if constexpr (rank <= 1) {
    return StorageOrder::row_major;
  } else {
    bool row_major = true;
    bool column_major = true;
    std::size_t right_stride = 1;
    std::size_t left_stride = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::size_t right_axis = rank - 1 - axis;
      row_major =
          row_major &&
          (static_cast<std::size_t>(view.stride(right_axis)) == right_stride);
      column_major =
          column_major &&
          (static_cast<std::size_t>(view.stride(axis)) == left_stride);
      right_stride *= extents[right_axis];
      left_stride *= extents[axis];
    }
    if (row_major) {
      return StorageOrder::row_major;
    }
    if (column_major) {
      return StorageOrder::column_major;
    }
    return std::nullopt;
  }
}

/**
 * @brief Storage order of a view that a dataset writes or reads as is.
 *
 * @tparam ViewType View type passed to a dataset
 * @param view View backing the dataset
 * @param name Dataset name, used in the error message
 * @return Storage order of the view's buffer
 * @throws std::invalid_argument if the buffer is not a contiguous LayoutLeft
 * or LayoutRight buffer
 */
template <typename ViewType>
StorageOrder require_contiguous_storage_order(const ViewType &view,
                                              const std::string &name) {
  const auto order = contiguous_storage_order(view);
  if (!order) {
    throw std::invalid_argument(
        "dataset " + name +
        ": view storage is not a contiguous LayoutLeft or LayoutRight buffer; "
        "pack tiled DomainView storage with "
        "specfem::io::impl::extract_sub_block or write "
        "specfem::datatype::storage_view()");
  }
  return *order;
}

/**
 * @brief Extents of @p view as they must be declared in a row-major file
 * format.
 *
 * Column-major views of rank >= 2 are reported with reversed extents because
 * their contiguous buffer is exactly the row-major array of the reversed
 * shape. Row-major views are reported as is.
 *
 * @tparam ViewType View type
 * @param view View whose extents are queried
 * @param order Storage order of the view's buffer
 * @return Extents to declare in the file
 */
template <typename ViewType>
std::vector<std::size_t> file_extents(const ViewType &view,
                                      const StorageOrder order) {
  constexpr int rank = ViewType::rank();
  std::vector<std::size_t> extents(rank);
  for (int i = 0; i < rank; ++i) {
    extents[i] = view.extent(i);
  }
  if (order == StorageOrder::column_major && rank >= 2) {
    std::reverse(extents.begin(), extents.end());
  }
  return extents;
}

} // namespace impl
} // namespace io_backends
} // namespace specfem
