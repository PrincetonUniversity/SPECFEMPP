#pragma once

#include "specfem/experimental/layout/domain_layout.hpp"
#include <string>
#include <type_traits>

/** @cond */
namespace specfem::experimental::layout_impl {

template <typename NativeView> struct KokkosStorage;

template <typename T> struct BorrowedStorage {
  using memory_space = Kokkos::HostSpace;
  using host_mirror_memory_space = Kokkos::HostSpace;
  using const_type = BorrowedStorage<const std::remove_const_t<T>>;
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
    require(offset < capacity, "layout flat offset out of bounds");
    return pointer[offset];
  }
  KOKKOS_INLINE_FUNCTION constexpr T &operator()(std::size_t offset) const {
    return access(offset);
  }
};

template <typename Mapping>
inline constexpr bool transformed_mapping = [] {
  if constexpr (requires { Mapping::is_transformed; }) {
    return Mapping::is_transformed;
  } else {
    return false;
  }
}();

template <typename Mapping>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
physical_span(const Mapping &mapping) {
  if constexpr (requires { mapping.physical_span_size(); }) {
    return mapping.physical_span_size();
  } else {
    return mapping.required_span_size();
  }
}

template <typename... Indices>
inline constexpr bool scalar_coordinates_v =
    ((!layout::is_tuple_v<Indices>) && ...);

template <typename Mapping, typename... Indices>
inline constexpr bool domain_extent_arguments_v =
    sizeof...(Indices) > 0 &&
    (!std::is_same_v<std::remove_cvref_t<Indices>, std::remove_cv_t<Mapping>> &&
     ...);

template <typename Storage>
concept AllocatingStorage =
    requires(const std::string &label, std::size_t count) {
      Storage::allocate(label, count);
    };

struct ViewAccess;

} // namespace specfem::experimental::layout_impl
/** @endcond */

namespace specfem::experimental::layout {

/**
 * @brief Backing storage interpreted through a standalone layout.
 *
 * Managed handles retain allocation ownership. Callers retain responsibility
 * for borrowed storage lifetime, memory-space access and asynchronous
 * completion. Const element types remain const; constness of the view object
 * alone does not make its elements const.
 * @tparam T Element type, possibly const.
 * @tparam Mapping Layout describing valid coordinates and element offsets.
 * @tparam Storage Internal backing handle, borrowed by default.
 * @tparam Transformed Whether a public transformation produced this view.
 */
template <typename T, typename Mapping,
          typename Storage = layout_impl::BorrowedStorage<T>,
          bool Transformed = false>
class View {
public:
  using value_type = T;
  using non_const_value_type = std::remove_const_t<T>;
  using memory_space = typename Storage::memory_space;
  using mapping_type = Mapping;
  /// Mutable host view, retaining borrowed semantics when storage is shared.
  using host_mirror_type = std::conditional_t<
      std::is_same_v<memory_space,
                     typename Storage::host_mirror_memory_space> &&
          !std::is_const_v<T>,
      View,
      View<non_const_value_type, Mapping,
           layout_impl::KokkosStorage<
               Kokkos::View<non_const_value_type *,
                            typename Storage::host_mirror_memory_space>>,
           Transformed>>;
  static constexpr bool is_physical_root =
      !Transformed && !layout_impl::transformed_mapping<Mapping>;

  /** @brief Construct an empty handle without allocation. */
  KOKKOS_INLINE_FUNCTION View() = default;

  /** @brief Allocate the mapping's full physical span.
   * @param label Kokkos allocation label. @param mapping Exact descriptor. */
  View(const std::string &label, const Mapping &mapping)
    requires layout_impl::AllocatingStorage<Storage>
      : View(Storage::allocate(label, layout_impl::physical_span(mapping)),
             mapping) {}

  /** @brief Allocate canonical Left or first-axis tiled Left storage.
   * @tparam Indices Logical extent types. @param label Allocation label.
   * @param indices Logical extents matching the mapping rank. */
  template <
      typename... Indices,
      std::enable_if_t<
          layout_impl::domain_extent_arguments_v<Mapping, Indices...>, int> = 0>
  View(const std::string &label, Indices... indices)
      : View(label, layout_impl::make_domain_layout_like<Mapping>(indices...)) {
  }

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
      const View<U, Mapping, OtherStorage, Transformed> &other)
      : storage_(other.storage_), layout_(other.layout_),
        initialized_(other.initialized_) {}

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
  operator()(const Coordinate &coordinate) const
    requires is_tuple_v<Coordinate>
  {
    return storage_.access(offset(coordinate));
  }

  /** @brief Access a valid scalar-indexed coordinate.
   * @tparam Indices Scalar coordinate types. @param indices Logical indices.
   * @return Element reference preserving constness. */
  template <
      typename... Indices,
      std::enable_if_t<layout_impl::scalar_coordinates_v<Indices...>, int> = 0>
  KOKKOS_INLINE_FUNCTION T &operator()(Indices... indices) const {
    return (*this)(layout::make_tuple(indices...));
  }

  /** @brief Coordinate rank. @return Number of top-level modes. */
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t rank() {
    return Mapping::rank();
  }

  /** @brief Inspect a static-axis extent. @tparam I Axis.
   * @return Extent, preserving static metadata. */
  template <std::size_t I>
  KOKKOS_INLINE_FUNCTION constexpr auto extent() const {
    layout_impl::require(initialized_, "layout uninitialized view");
    return layout_.template extent<I>();
  }

  /** @brief Inspect an extent. @param axis Axis. @return Logical extent. */
  KOKKOS_INLINE_FUNCTION std::size_t extent(std::size_t axis) const {
    layout_impl::require(initialized_, "layout uninitialized view");
    return layout_impl::runtime_extent(layout_, axis);
  }

  /** @brief Test membership. @tparam Coordinate Tuple type.
   * @param coordinate Coordinate to test. @return Whether access is valid. */
  template <typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr bool
  valid(const Coordinate &coordinate) const
    requires is_tuple_v<Coordinate>
  {
    return initialized_ && layout_.valid(coordinate);
  }
  /** @brief Test scalar membership. @tparam Indices Scalar types.
   * @param indices Logical indices. @return Whether access is valid. */
  template <
      typename... Indices,
      std::enable_if_t<layout_impl::scalar_coordinates_v<Indices...>, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr bool valid(Indices... indices) const {
    return valid(layout::make_tuple(indices...));
  }

  /** @brief Original-pointer-relative offset. @tparam Coordinate Tuple type.
   * @param coordinate Valid coordinate. @return Element offset. */
  template <typename Coordinate>
  KOKKOS_INLINE_FUNCTION constexpr std::size_t
  offset(const Coordinate &coordinate) const
    requires is_tuple_v<Coordinate>
  {
    layout_impl::require(initialized_, "layout uninitialized view");
    return layout_.offset(coordinate);
  }
  /** @brief Scalar coordinate offset. @tparam Indices Scalar types.
   * @param indices Valid logical indices. @return Element offset. */
  template <
      typename... Indices,
      std::enable_if_t<layout_impl::scalar_coordinates_v<Indices...>, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr std::size_t
  offset(Indices... indices) const {
    return offset(layout::make_tuple(indices...));
  }

  /** @brief Retain flat storage access.
   * @return Backing adapter preserving element constness. */
  KOKKOS_INLINE_FUNCTION auto flat_storage() const {
    if constexpr (std::is_const_v<T>) {
      return typename Storage::const_type(storage_);
    } else {
      return storage_;
    }
  }
  /** @brief Declared backing region. @return Number of storage elements. */
  KOKKOS_INLINE_FUNCTION std::size_t storage_span_size() const {
    return storage_.size();
  }
  /** @brief Original physical span. @return Full allocation requirement. */
  KOKKOS_INLINE_FUNCTION std::size_t physical_span_size() const {
    return initialized_ ? layout_impl::physical_span(layout_) : 0;
  }

private:
  friend struct layout_impl::ViewAccess;
  template <typename, typename, typename, bool> friend class View;

  KOKKOS_INLINE_FUNCTION constexpr View(const Storage &storage, Mapping mapping)
      : storage_(storage), layout_(mapping), initialized_(true) {
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
    return View<T, NewMapping, Storage, true>(storage_, mapping);
  }

  Storage storage_{};
  Mapping layout_{};
  bool initialized_ = false;
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
