#pragma once

#include "specfem/experimental/layout/layout.hpp"
#include <type_traits>

/** @cond */
namespace specfem::experimental::layout_impl {

// Ownership adapters can retain their original handle here in a later ticket.
template <typename T> struct BorrowedStorage {
  T *pointer = nullptr;
  std::size_t capacity = 0;
};

struct ViewAccess;

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Borrowed storage interpreted through a standalone layout.
 *
 * The caller retains responsibility for storage lifetime, memory-space access,
 * and completion of asynchronous work. Const element types remain const;
 * constness of the view object alone does not make its elements const.
 * @tparam T Element type, possibly const.
 * @tparam Mapping Layout describing valid coordinates and element offsets.
 */
template <typename T, typename Mapping> class View {
public:
  /**
   * @brief Associate a borrowed allocation with a mapping.
   * @param data Original backing pointer, null only for an empty mapping.
   * @param capacity Accessible element capacity measured from data.
   * @param mapping Mapping whose required span must fit capacity.
   */
  KOKKOS_INLINE_FUNCTION constexpr View(T *data, std::size_t capacity,
                                        Mapping mapping)
      : storage_{ data, capacity }, layout_(mapping) {
    layout_impl::require(layout_.required_span_size() <= capacity,
                         "layout exceeds backing capacity");
    layout_impl::require(data != nullptr || layout_.empty(),
                         "layout requires nonnull backing storage");
  }

  /**
   * @brief Convert from a view whose elements convert to T, such as a mutable
   * view to a read-only one.
   *
   * Array-pointer convertibility admits added const but rejects
   * derived-to-base conversions, which would break element arithmetic.
   * @tparam U Source element type.
   * @param other View sharing this mapping type and the same backing storage.
   */
  template <typename U>
    requires std::is_convertible_v<U (*)[], T (*)[]>
  KOKKOS_INLINE_FUNCTION constexpr View(const View<U, Mapping> &other)
      : storage_{ other.storage_.pointer, other.storage_.capacity },
        layout_(other.layout_) {}

  /** @brief Inspect the mapping. @return Original coordinate layout. */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr const Mapping &layout() const {
    return layout_;
  }

  /**
   * @brief Access one valid element without changing storage ownership.
   * @tparam Coordinate Coordinate tuple accepted by the layout.
   * @param coordinate Valid logical coordinate.
   * @return Reference preserving element constness.
   */
  template <typename Coordinate>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr T &
  operator()(const Coordinate &coordinate) const {
    return storage_.pointer[layout_.offset(coordinate)];
  }

private:
  friend struct layout_impl::ViewAccess;
  template <typename, typename> friend class View;

  // Only layout transformations should rewrap storage. They must establish
  // parent-domain correspondence and retain pointer-relative displacement.
  template <typename NewMapping>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
  rebind_layout(const NewMapping &mapping) const {
    return View<T, NewMapping>(storage_.pointer, storage_.capacity, mapping);
  }

  layout_impl::BorrowedStorage<T> storage_{};
  Mapping layout_;
};

/**
 * @brief Deduce a borrowed view without allocating or copying elements.
 * @tparam T Backing element type, possibly const.
 * @tparam Mapping Layout type.
 * @param data Original backing pointer in caller-accessible memory.
 * @param capacity Number of accessible elements, including allocation padding.
 * @param mapping Coordinate mapping with required span no greater than
 * capacity.
 * @return Borrowed view retaining the pointer and capacity.
 */
template <typename T, typename Mapping>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
make_view(T *data, std::size_t capacity, const Mapping &mapping) {
  return View<T, Mapping>(data, capacity, mapping);
}

} // namespace specfem::experimental::layout

/** @cond */
namespace specfem::experimental::layout_impl {

// The single propagation operation shared by future layout transformations.
struct ViewAccess {
  template <typename View, typename Mapping>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr auto
  rebind_layout(const View &view, const Mapping &mapping) {
    return view.rebind_layout(mapping);
  }
};

} // namespace specfem::experimental::layout_impl
/** @endcond */
