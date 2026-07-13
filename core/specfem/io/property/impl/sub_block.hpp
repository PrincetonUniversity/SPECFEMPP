#pragma once

#include <cstddef>
#include <string>

namespace specfem {
namespace io {
namespace property_impl {

/**
 * @brief Allocate a host view shaped like a leading-dimension sub-block
 *        [count][ngll...] of @p view.
 *
 * Property views span a whole (medium, property) group while the property
 * writer/reader emit one file group per (medium, property, attenuation)
 * combination; datasets of a combination therefore cover only a contiguous
 * leading-dimension slice of the property view. The chunk-tiled domain-view
 * storage makes such a slice non-contiguous in memory, so sub-block I/O goes
 * through a correctly-shaped scratch view instead of a subview.
 *
 * @tparam ViewType Host property view type (rank 3 or 4)
 * @param view View whose trailing (GLL) extents are replicated
 * @param name Label prefix for the scratch allocation
 * @param count Leading-dimension (element) extent of the sub-block
 * @return Freshly allocated, uninitialized sub-block view
 */
template <typename ViewType>
ViewType make_sub_block(const ViewType &view, const std::string &name,
                        const int count) {
  static_assert(ViewType::rank() == 3 || ViewType::rank() == 4,
                "sub-block helpers expect element-major per-GLL views");
  if constexpr (ViewType::rank() == 3) {
    return ViewType(name + "_sub", static_cast<std::size_t>(count),
                    view.extent(1), view.extent(2));
  } else {
    return ViewType(name + "_sub", static_cast<std::size_t>(count),
                    view.extent(1), view.extent(2), view.extent(3));
  }
}

/**
 * @brief Copy the leading-dimension slice [offset, offset + count) of
 *        @p view into a fresh sub-block view.
 *
 * Returns @p view unchanged when the slice spans the whole view (single
 * attenuation tag in the group -- the common case), so no copy happens.
 *
 * @tparam ViewType Host property view type (rank 3 or 4)
 * @param view Source view (group-local indexing)
 * @param name Label prefix for the scratch allocation
 * @param offset First element of the slice
 * @param count Number of elements in the slice
 * @return View holding the slice values
 */
template <typename ViewType>
ViewType extract_sub_block(const ViewType &view, const std::string &name,
                           const int offset, const int count) {
  if (offset == 0 && count == static_cast<int>(view.extent(0)))
    return view;
  ViewType sub = make_sub_block(view, name, count);
  if constexpr (ViewType::rank() == 3) {
    for (int e = 0; e < count; ++e)
      for (std::size_t iz = 0; iz < view.extent(1); ++iz)
        for (std::size_t ix = 0; ix < view.extent(2); ++ix)
          sub(e, iz, ix) = view(offset + e, iz, ix);
  } else {
    for (int e = 0; e < count; ++e)
      for (std::size_t iz = 0; iz < view.extent(1); ++iz)
        for (std::size_t iy = 0; iy < view.extent(2); ++iy)
          for (std::size_t ix = 0; ix < view.extent(3); ++ix)
            sub(e, iz, iy, ix) = view(offset + e, iz, iy, ix);
  }
  return sub;
}

/**
 * @brief Copy a sub-block view into @p dst at leading-dimension @p offset.
 *
 * @tparam ViewType Host property view type (rank 3 or 4)
 * @param dst Destination view (group-local indexing)
 * @param src Sub-block view holding the values to insert
 * @param offset First destination element of the slice
 */
template <typename ViewType>
void insert_sub_block(const ViewType &dst, const ViewType &src,
                      const int offset) {
  const int count = static_cast<int>(src.extent(0));
  if constexpr (ViewType::rank() == 3) {
    for (int e = 0; e < count; ++e)
      for (std::size_t iz = 0; iz < src.extent(1); ++iz)
        for (std::size_t ix = 0; ix < src.extent(2); ++ix)
          dst(offset + e, iz, ix) = src(e, iz, ix);
  } else {
    for (int e = 0; e < count; ++e)
      for (std::size_t iz = 0; iz < src.extent(1); ++iz)
        for (std::size_t iy = 0; iy < src.extent(2); ++iy)
          for (std::size_t ix = 0; ix < src.extent(3); ++ix)
            dst(offset + e, iz, iy, ix) = src(e, iz, iy, ix);
  }
}

} // namespace property_impl
} // namespace io
} // namespace specfem
