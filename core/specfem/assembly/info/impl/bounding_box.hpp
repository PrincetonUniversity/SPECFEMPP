#pragma once
#include "bounds.hpp"
#include "specfem/datatype/impl/register_array.hpp"
#include "enumerations/interface.hpp"

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::info::impl {
template <specfem::dimension::type DimensionTag>
struct BoundingBox {

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static int ndim =
      specfem::dimension::dimension<dimension_tag>::dim; ///< Number of dimensions

  specfem::datatype::impl::RegisterArray<
    Bounds, 
    Kokkos::extents<size_t, ndim>, 
    Kokkos::layout_left> bounds_array;

  BoundingBox() = default;

  template <specfem::dimension::type U = dimension_tag,
          typename std::enable_if<U == specfem::dimension::type::dim2>::type
              * = nullptr>
  BoundingBox(const type_real x_min, const type_real x_max, const type_real z_min,
              const type_real z_max)
      : bounds_array(Bounds(x_min, x_max), Bounds(z_min, z_max)) {};

  template <specfem::dimension::type U = dimension_tag,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type
              * = nullptr>
  BoundingBox(const type_real x_min, const type_real x_max, const type_real y_min,
              const type_real y_max, const type_real z_min, const type_real z_max)
      : bounds_array(Bounds(x_min, x_max), Bounds(y_min, y_max), Bounds(z_min, z_max)) {};

  template <specfem::dimension::type U = dimension_tag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  BoundingBox(const std::vector<Bounds> &bounds) {
    if (bounds.size() != 2) {
      throw std::invalid_argument(
          "BoundingBox<dim2> requires a vector of size 2");
    }
    bounds_array = decltype(bounds_array)(bounds[0], bounds[1]);
  }

  template <specfem::dimension::type U = dimension_tag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  BoundingBox(const std::vector<Bounds> &bounds) {
    if (bounds.size() != 3) {
      throw std::invalid_argument(
          "BoundingBox<dim3> requires a vector of size 3");
    }
    bounds_array = decltype(bounds_array)(bounds[0], bounds[1], bounds[2]);
  }

  Bounds &x() {
    return bounds_array(0);
  }

  const Bounds &x() const {
    return bounds_array(0);
  }

  template <specfem::dimension::type U = dimension_tag,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type
              * = nullptr>
  Bounds &y() {
    return bounds_array(1);
  }

  template <specfem::dimension::type U = dimension_tag,
          typename std::enable_if<U == specfem::dimension::type::dim3>::type
              * = nullptr>
  const Bounds &y() const {
    return bounds_array(1);
  }

  Bounds &z() {
    if constexpr (dimension_tag == specfem::dimension::type::dim2) {
      return bounds_array(1);
    } else {
      return bounds_array(2);
    }
  }

  const Bounds &z() const {
    if constexpr (dimension_tag == specfem::dimension::type::dim2) {
      return bounds_array(1);
    } else {
      return bounds_array(2);
    }
  }

};

} // namespace specfem::assembly::info::impl
