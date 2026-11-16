#pragma once

#include "specfem/experimental/layout/layout.hpp"
#include <type_traits>

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename T> struct BorrowedStorage {
  KOKKOS_INLINE_FUNCTION constexpr BorrowedStorage() = default;

  KOKKOS_INLINE_FUNCTION constexpr BorrowedStorage(T *data, std::size_t size)
      : pointer(data), capacity(size) {}

  template <typename U>
    requires std::is_convertible_v<U (*)[], T (*)[]>
  KOKKOS_INLINE_FUNCTION constexpr BorrowedStorage(
      const BorrowedStorage<U> &other)
      : pointer(other.pointer), capacity(other.capacity) {}

  T *pointer = nullptr;
  std::size_t capacity = 0;

  KOKKOS_INLINE_FUNCTION constexpr T *data() const { return pointer; }
  KOKKOS_INLINE_FUNCTION constexpr std::size_t size() const { return capacity; }
  KOKKOS_INLINE_FUNCTION constexpr T &access(std::size_t offset) const {
    return pointer[offset];
  }
};

struct ViewAccess;

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Backing storage interpreted through a standalone layout.
 *
 * The caller retains responsibility for storage lifetime, memory-space access,
 * and completion of asynchronous work. Const element types remain const;
 * constness of the view object alone does not make its elements const.
 * @tparam T Element type, possibly const.
 * @tparam Mapping Layout describing valid coordinates and element offsets.
 * @tparam Storage Internal backing handle, borrowed by default.
 */
template <typename T, typename Mapping,
          typename Storage = layout_impl::BorrowedStorage<T>>
class View {
public:
  /**
   * @brief Associate a borrowed allocation with a mapping.
   * @param data Original backing pointer, null only for an empty mapping.
   * @param capacity Accessible element capacity measured from data.
   * @param mapping Mapping whose required span must fit capacity.
   */
  KOKKOS_INLINE_FUNCTION constexpr View(T *data, std::size_t capacity,
                                        Mapping mapping)
    requires std::is_same_v<Storage, layout_impl::BorrowedStorage<T>>
      : View(Storage{ data, capacity }, mapping) {}

  /**
   * @brief Convert from a view whose elements convert to T, such as a mutable
   * view to a read-only one.
   *
   * Array-pointer convertibility admits added const but rejects
   * derived-to-base conversions, which would break element arithmetic.
   * @tparam U Source element type.
   * @tparam OtherStorage Source backing handle, convertible to Storage.
   * @param other View sharing this mapping type and the same backing storage.
   */
  template <typename U, typename OtherStorage>
    requires(std::is_convertible_v<U (*)[], T (*)[]> &&
             std::is_convertible_v<OtherStorage, Storage>)
  KOKKOS_INLINE_FUNCTION constexpr View(
      const View<U, Mapping, OtherStorage> &other)
      : storage_(other.storage_), layout_(other.layout_) {}

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
    return storage_.access(layout_.offset(coordinate));
  }

private:
  friend struct layout_impl::ViewAccess;
  template <typename, typename, typename> friend class View;

  KOKKOS_INLINE_FUNCTION constexpr View(const Storage &storage, Mapping mapping)
      : storage_(storage), layout_(mapping) {
    layout_impl::require(layout_.required_span_size() <= storage_.size(),
                         "layout exceeds backing capacity");
    layout_impl::require(storage_.data() != nullptr || layout_.empty(),
                         "layout requires nonnull backing storage");
  }

  // Only layout transformations should rewrap storage. They must establish
  // parent-domain correspondence and retain pointer-relative displacement.
  template <typename NewMapping>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr auto
  rebind_layout(const NewMapping &mapping) const {
    return View<T, NewMapping, Storage>(storage_, mapping);
  }

  Storage storage_;
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
  template <typename T, typename Storage, typename Mapping>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr auto
  make_view(const Storage &storage, const Mapping &mapping) {
    return layout::View<T, Mapping, Storage>(storage, mapping);
  }

  template <typename View, typename Mapping>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr auto
  rebind_layout(const View &view, const Mapping &mapping) {
    return view.rebind_layout(mapping);
  }
};

} // namespace specfem::experimental::layout_impl
/** @endcond */
