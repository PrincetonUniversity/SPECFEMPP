#pragma once

#include "specfem/datatype/point_view.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

namespace specfem::datatype::impl {

/**
 * @brief Vector values at one face point of a chunk-face view
 *
 * Returned by `VectorChunkFaceViewType::operator()(index)`. Reads and writes
 * the components stored at `(index.iface, index.ipoint_i, index.ipoint_j)` of
 * the parent view, so that a point view or a point field is stored in one
 * statement:
 *
 * @code
 * flux(local_index) = point_flux;
 * @endcode
 *
 * The index must be local to the chunk: `iface` is the position of the face
 * within the chunk. `ispec` is not read.
 *
 * The subview holds references to the parent view and to the index and must
 * not outlive either.
 *
 * @tparam ViewType The type of the parent view this subview accesses
 */
template <typename ViewType> struct VectorChunkFaceSubview {
  /// Index type from the parent view
  using index_type = typename ViewType::index_type;
  /// Point view type for component access
  using point_view_type = typename ViewType::point_view_type;

  /// Reference to the parent view
  ViewType &view;
  /// Reference to the index within the parent view
  const index_type &index;

  /**
   * @brief Construct a subview of one face point
   *
   * @param view Reference to the parent view
   * @param index Reference to the chunk-local face point index
   */
  KOKKOS_INLINE_FUNCTION
  VectorChunkFaceSubview(ViewType &view, const index_type &index)
      : view(view), index(index) {}

  /**
   * @brief Access a component of the vector (non-const)
   *
   * @param icomp Component index to access
   * @return Reference to the component value
   */
  KOKKOS_INLINE_FUNCTION constexpr auto &operator()(const int &icomp) {
    return view(index.iface, index.ipoint_i, index.ipoint_j, icomp);
  }

  /**
   * @brief Access a component of the vector (const)
   *
   * @param icomp Component index to access
   * @return Value of the component
   */
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const int &icomp) const {
    return view(index.iface, index.ipoint_i, index.ipoint_j, icomp);
  }

  /**
   * @brief Store a point view at this face point
   *
   * Copies all components from the source point view to this subview.
   *
   * @param other Source point view to copy from
   * @return Reference to this subview after assignment
   */
  KOKKOS_INLINE_FUNCTION
  auto &operator=(const point_view_type &other) {
    for (int icomp = 0; icomp < point_view_type::components; ++icomp) {
      (*this)(icomp) = other(icomp);
    }
    return *this;
  }

  /**
   * @brief Store a point field at this face point
   *
   * Copies all components of a point field, e.g.
   * `specfem::point::acceleration`, whose data is this view's point view type.
   *
   * @tparam PointFieldType Point field whose `get_data()` returns
   * `point_view_type`
   * @param field Source point field to copy from
   * @return Reference to this subview after assignment
   */
  template <typename PointFieldType,
            typename std::enable_if_t<
                std::is_same_v<
                    std::decay_t<decltype(std::declval<const PointFieldType &>()
                                              .get_data())>,
                    point_view_type>,
                int> = 0>
  KOKKOS_INLINE_FUNCTION auto &operator=(const PointFieldType &field) {
    return (*this = field.get_data());
  }
};

} // namespace specfem::datatype::impl
