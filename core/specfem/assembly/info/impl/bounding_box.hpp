#pragma once
#include "bounds.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::info::impl {

/**
 * @brief Axis-aligned bounding box for mesh domain.
 *
 * Stores min/max bounds for each spatial dimension (x, y, z).
 * Provides dimension-specific accessors that handle 2D vs 3D cases.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> struct BoundingBox {

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static int ndim =
      specfem::element::dimension<dimension_tag>::dim; ///< Number of dimensions

  specfem::datatype::RegisterArray<Bounds, Kokkos::extents<size_t, ndim>,
                                   Kokkos::layout_left>
      bounds_array; ///< Array of bounds per dimension

  BoundingBox() = default;

  /**
   * @brief Construct 2D bounding box from explicit min/max values.
   * @param x_min Minimum x coordinate
   * @param x_max Maximum x coordinate
   * @param z_min Minimum z coordinate
   * @param z_max Maximum z coordinate
   */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  BoundingBox(const type_real x_min, const type_real x_max,
              const type_real z_min, const type_real z_max)
      : bounds_array(Bounds(x_min, x_max), Bounds(z_min, z_max)){};

  /**
   * @brief Construct 3D bounding box from explicit min/max values.
   * @param x_min Minimum x coordinate
   * @param x_max Maximum x coordinate
   * @param y_min Minimum y coordinate
   * @param y_max Maximum y coordinate
   * @param z_min Minimum z coordinate
   * @param z_max Maximum z coordinate
   */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  BoundingBox(const type_real x_min, const type_real x_max,
              const type_real y_min, const type_real y_max,
              const type_real z_min, const type_real z_max)
      : bounds_array(Bounds(x_min, x_max), Bounds(y_min, y_max),
                     Bounds(z_min, z_max)){};

  /**
   * @brief Construct 2D bounding box from vector of Bounds.
   * @param bounds Vector of 2 Bounds objects (x, z)
   * @throws std::invalid_argument if bounds.size() != 2
   */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  BoundingBox(const std::vector<Bounds> &bounds) {
    if (bounds.size() != 2) {
      throw std::invalid_argument(
          "BoundingBox<dim2> requires a vector of size 2");
    }
    bounds_array = decltype(bounds_array)(bounds[0], bounds[1]);
  }

  /**
   * @brief Construct 3D bounding box from vector of Bounds.
   * @param bounds Vector of 3 Bounds objects (x, y, z)
   * @throws std::invalid_argument if bounds.size() != 3
   */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  BoundingBox(const std::vector<Bounds> &bounds) {
    if (bounds.size() != 3) {
      throw std::invalid_argument(
          "BoundingBox<dim3> requires a vector of size 3");
    }
    bounds_array = decltype(bounds_array)(bounds[0], bounds[1], bounds[2]);
  }

  /** @brief Access x-direction bounds. */
  Bounds &x() { return bounds_array(0); }

  /** @brief Access x-direction bounds (const). */
  const Bounds &x() const { return bounds_array(0); }

  /** @brief Access y-direction bounds (3D only). */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  Bounds &y() {
    return bounds_array(1);
  }

  /** @brief Access y-direction bounds (3D only, const). */
  template <specfem::element::dimension_tag U = dimension_tag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  const Bounds &y() const {
    return bounds_array(1);
  }

  /** @brief Access z-direction bounds (index 1 for 2D, index 2 for 3D). */
  Bounds &z() {
    if constexpr (dimension_tag == specfem::element::dimension_tag::dim2) {
      return bounds_array(1);
    } else {
      return bounds_array(2);
    }
  }

  /** @brief Access z-direction bounds (const, index 1 for 2D, index 2 for 3D).
   */
  const Bounds &z() const {
    if constexpr (dimension_tag == specfem::element::dimension_tag::dim2) {
      return bounds_array(1);
    } else {
      return bounds_array(2);
    }
  }
};

} // namespace specfem::assembly::info::impl
