#pragma once

#include "enumerations/dimension.hpp"
#include "specfem/datatype/point_view.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::datatype::impl {

/**
 * @brief Subview for accessing vector components within a chunk element
 *
 * This class provides a convenient interface for accessing vector data
 * components at a specific index within a larger multi-dimensional view.
 * It simplifies operations by abstracting the full indexing requirements.
 *
 * @tparam ViewType The type of the parent view this subview accesses
 */
template <typename ViewType> struct VectorChunkElementSubview {
  /// Index type from the parent view
  using index_type = typename ViewType::index_type;
  /// Point view type for component access
  using point_view_type = typename ViewType::point_view_type;

  /// Reference to the parent view
  ViewType &view;
  /// Reference to the index within the parent view
  const index_type &index;

  /**
   * @brief Construct a new Vector Chunk Subview
   *
   * @param view Reference to the parent view
   * @param index Reference to the index within the parent view
   */
  KOKKOS_INLINE_FUNCTION
  VectorChunkElementSubview(ViewType &view, const index_type &index)
      : view(view), index(index) {}

  /**
   * @brief Access a specific component of the vector (non-const)
   *
   * Specialized for 2D case, providing element access with proper indexing.
   *
   * @param icomp Component index to access
   * @return Reference to the component value
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto &operator()(const int &icomp) {
    return view(index.ispec, index.iz, index.ix, icomp);
  }

  /**
   * @brief Access a specific component of the vector (non-const)
   *
   * Specialized for 3D case, providing element access with proper indexing.
   *
   * @param icomp Component index to access
   * @return Reference to the component value
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto &operator()(const int &icomp) {
    return view(index.ispec, index.iz, index.iy, index.ix, icomp);
  }

  /**
   * @brief Access a specific component of the vector (const)
   *
   * Specialized for 2D case, providing const element access with proper
   * indexing.
   *
   * @param icomp Component index to access
   * @return Value of the component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const int &icomp) const {
    return view(index.ispec, index.iz, index.ix, icomp);
  }

  /**
   * @brief Access a specific component of the vector (const)
   *
   * Specialized for 3D case, providing const element access with proper
   * indexing.
   *
   * @param icomp Component index to access
   * @return Value of the component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const int &icomp) const {
    return view(index.ispec, index.iz, index.iy, index.ix, icomp);
  }

  /**
   * @brief Assignment operator from another point view
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
};

/**
 * @brief Subview for accessing tensor components within a chunk element
 *
 * This class provides a convenient interface for accessing tensor data
 * at a specific index within a larger multi-dimensional view.
 * It handles the proper indexing based on the dimensionality of the problem.
 *
 * @tparam ViewType The type of the parent view this subview accesses
 */
template <typename ViewType> struct TensorChunkElementSubview {
  /// Index type from the parent view
  using index_type = typename ViewType::index_type;
  /// Point view type for tensor component access
  using point_view_type = typename ViewType::point_view_type;

  /// Reference to the parent view
  ViewType &view;
  /// Reference to the index within the parent view
  const index_type &index;

  /**
   * @brief Get the rank of the tensor
   *
   * @return Rank of the tensor (always 2 for this implementation)
   */
  KOKKOS_INLINE_FUNCTION
  constexpr static std::size_t rank() { return 2; }

  /**
   * @brief Get the extent of a dimension
   *
   * @param i Dimension index
   * @return Size of the dimension
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr std::size_t extent(const size_t &i) const {
    return view.extent(i + 3);
  }

  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr std::size_t extent(const size_t &i) const {
    return view.extent(i + 4);
  }

  /**
   * @brief Get the static extent of a dimension
   *
   * @param i Dimension index
   * @return Static size of the dimension
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr static std::size_t
  static_extent(const size_t &i) {
    return ViewType::static_extent(i + 3);
  }

  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr static std::size_t
  static_extent(const size_t &i) {
    return ViewType::static_extent(i + 4);
  }

  /**
   * @brief Construct a new Tensor Chunk Subview
   *
   * @param view Reference to the parent view
   * @param index Reference to the index within the parent view
   */
  KOKKOS_INLINE_FUNCTION
  TensorChunkElementSubview(ViewType &view, const index_type &index)
      : view(view), index(index) {}

  /**
   * @brief Access a specific tensor component (non-const)
   *
   * Specialized for 2D case, providing element access with proper indexing.
   *
   * @param icomp Component index
   * @param idim Dimension index
   * @return Reference to the tensor component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto &operator()(const int &icomp,
                                                    const int &idim) {
    return view(index.ispec, index.iz, index.ix, icomp, idim);
  }

  /**
   * @brief Access a specific tensor component (non-const)
   *
   * Specialized for 3D case, providing element access with proper indexing.
   *
   * @param icomp Component index
   * @param idim Dimension index
   * @return Reference to the tensor component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto &operator()(const int &icomp,
                                                    const int &idim) {
    return view(index.ispec, index.iz, index.iy, index.ix, icomp, idim);
  }

  /**
   * @brief Access a specific tensor component (const)
   *
   * Specialized for 2D case, providing const element access with proper
   * indexing.
   *
   * @param icomp Component index
   * @param idim Dimension index
   * @return Value of the tensor component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const int &icomp,
                                                   const int &idim) const {
    return view(index.ispec, index.iz, index.ix, icomp, idim);
  }

  /**
   * @brief Access a specific tensor component (const)
   *
   * Specialized for 3D case, providing const element access with proper
   * indexing.
   *
   * @param icomp Component index
   * @param idim Dimension index
   * @return Value of the tensor component
   */
  template <
      specfem::dimension::type D = index_type::dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr auto operator()(const int &icomp,
                                                   const int &idim) const {
    return view(index.ispec, index.iz, index.iy, index.ix, icomp, idim);
  }

  /**
   * @brief Assignment operator from another point view
   *
   * Copies all tensor components from the source point view to this subview.
   *
   * @param other Source point view to copy from
   * @return Reference to this subview after assignment
   */
  KOKKOS_INLINE_FUNCTION
  auto &operator=(const point_view_type &other) {
    for (int icomp = 0; icomp < point_view_type::components; ++icomp) {
      for (int idim = 0; idim < point_view_type::dimensions; ++idim) {
        (*this)(icomp, idim) = other(icomp, idim);
      }
    }
    return *this;
  }
};

} // namespace specfem::datatype::impl
