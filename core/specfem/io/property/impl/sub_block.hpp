#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <string>
#include <type_traits>

namespace specfem {
namespace io {
namespace property_impl {

/**
 * @brief Allocate a plain row-major host view shaped like a leading-dimension
 *        sub-block [count][ngll...] of @p view.
 *
 * Property and attenuation-model views are chunk-tiled domain views: their
 * flat storage interleaves elements in SIMD-width tiles, padded to a full
 * tile at the tail. The I/O backends serialize a view's flat storage
 * linearly, so writing a domain view directly would (a) truncate real values
 * interleaved into the padded tail whenever the element count is not a
 * multiple of the tile width, and (b) produce a file whose payload order
 * depends on the build's SIMD width. Every dataset is therefore staged
 * through a plain LayoutRight view, making the file hold the logical
 * row-major values independent of tiling.
 *
 * @tparam ViewType Domain view type (rank 3 or 4)
 * @param view View whose trailing (GLL) extents are replicated
 * @param name Label prefix for the scratch allocation
 * @param count Leading-dimension (element) extent of the sub-block
 * @return Plain LayoutRight host view of shape [count][ngll...]
 */
template <typename ViewType>
auto make_sub_block(const ViewType &view, const std::string &name,
                    const int count) {
  static_assert(ViewType::rank() == 3 || ViewType::rank() == 4,
                "sub-block helpers expect element-major per-GLL views");
  using value_type = std::remove_const_t<typename ViewType::value_type>;
  if constexpr (ViewType::rank() == 3) {
    return Kokkos::View<value_type ***, Kokkos::LayoutRight, Kokkos::HostSpace>(
        name + "_sub", static_cast<std::size_t>(count), view.extent(1),
        view.extent(2));
  } else {
    return Kokkos::View<value_type ****, Kokkos::LayoutRight,
                        Kokkos::HostSpace>(
        name + "_sub", static_cast<std::size_t>(count), view.extent(1),
        view.extent(2), view.extent(3));
  }
}

/**
 * @brief Pack the leading-dimension slice [offset, offset + count) of
 *        @p view into a plain row-major host view for serialization.
 *
 * @tparam ViewType Domain view type (rank 3 or 4)
 * @param view Source domain view (group-local indexing)
 * @param name Label prefix for the scratch allocation
 * @param offset First element of the slice
 * @param count Number of elements in the slice
 * @return Plain LayoutRight host view holding the slice values
 */
template <typename ViewType>
auto extract_sub_block(const ViewType &view, const std::string &name,
                       const int offset, const int count) {
  auto sub = make_sub_block(view, name, count);
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
 * @brief Unpack a plain row-major sub-block view into the domain view @p dst
 *        at leading-dimension @p offset.
 *
 * @tparam ViewType Destination domain view type (rank 3 or 4)
 * @tparam PlainViewType Plain LayoutRight host view type from make_sub_block
 * @param dst Destination domain view (group-local indexing)
 * @param src Sub-block view holding the values to insert
 * @param offset First destination element of the slice
 */
template <typename ViewType, typename PlainViewType>
void insert_sub_block(const ViewType &dst, const PlainViewType &src,
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
