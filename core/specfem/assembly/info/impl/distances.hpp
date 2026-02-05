#pragma once
#include "specfem/element.hpp"
#include "specfem/assembly/mesh.hpp"

namespace specfem::assembly::info::impl {


/// @brief Compute distances between adjacent GLL points in all directions
/// @tparam DimensionTag The dimension type (dim2 or dim3)
/// @note Y-direction is only computed for dim3 using if constexpr
template <specfem::element::dimension_tag DimensionTag,
          typename GllDistanceAcc, typename ElementGllDistanceAcc>
KOKKOS_INLINE_FUNCTION void compute_gll_distances(
    const specfem::point::index<DimensionTag> &point_index, const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::point::global_coordinates<DimensionTag> &current_point,
    GllDistanceAcc &gll_distance_acc,
    ElementGllDistanceAcc &element_gll_distance_acc) {

  constexpr static specfem::element::dimension_tag dimension_tag = DimensionTag;

  const int ngll_minus_1 = mesh.element_grid.ngll - 1;

  // X direction
  if (point_index.ix < ngll_minus_1) {
    specfem::point::index<dimension_tag> next_index = point_index;
    specfem::point::global_coordinates<dimension_tag> next_point;
    next_index.ix += 1;
    specfem::assembly::load_on_device(next_index, mesh, next_point);
    type_real dist = specfem::point::distance(current_point, next_point);
    gll_distance_acc.update(dist);
    element_gll_distance_acc.update(point_index.ispec, dist);
  }

  // Y direction (dim3 only)
  if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
    if (point_index.iy < ngll_minus_1) {
      specfem::point::index<dimension_tag> next_index = point_index;
      specfem::point::global_coordinates<dimension_tag> next_point;
      next_index.iy += 1;
      specfem::assembly::load_on_device(next_index, mesh, next_point);
      type_real dist = specfem::point::distance(current_point, next_point);
      gll_distance_acc.update(dist);
      element_gll_distance_acc.update(point_index.ispec, dist);
    }
  }

  // Z direction
  if (point_index.iz < ngll_minus_1) {
    specfem::point::index<dimension_tag> next_index = point_index;
    specfem::point::global_coordinates<dimension_tag> next_point;
    next_index.iz += 1;
    specfem::assembly::load_on_device(next_index, mesh, next_point);
    type_real dist = specfem::point::distance(current_point, next_point);
    gll_distance_acc.update(dist);
    element_gll_distance_acc.update(point_index.ispec, dist);
  }
}

/// @brief Compute element sizes (corner-to-corner distances) in all directions
/// @tparam DimensionTag The dimension type (dim2 or dim3)
/// @note Y-direction is only computed for dim3 using if constexpr
template <specfem::element::dimension_tag DimensionTag,
          typename DistanceAcc, typename ElementDistanceAcc>
KOKKOS_INLINE_FUNCTION void compute_element_sizes(
    const specfem::point::index<DimensionTag> &point_index, const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::point::global_coordinates<DimensionTag> &current_point,
    DistanceAcc &distance_acc, ElementDistanceAcc &element_distance_acc) {

  constexpr static specfem::element::dimension_tag dimension_tag = DimensionTag;

  const int fgll = mesh.element_grid.ngll - 1;

  // X direction
  if (point_index.ix == 0) {
    specfem::point::index<DimensionTag> corner_index = point_index;
    specfem::point::global_coordinates<dimension_tag> corner_point;
    corner_index.ix += fgll;
    specfem::assembly::load_on_device(corner_index, mesh, corner_point);
    type_real dist = specfem::point::distance(current_point, corner_point);
    distance_acc.update(dist);
    element_distance_acc.update(point_index.ispec, dist);
  }

  // Y direction (dim3 only)
  if constexpr (dimension_tag == specfem::element::dimension_tag::dim3) {
    if (point_index.iy == 0) {
      specfem::point::index<dimension_tag> corner_index = point_index;
      specfem::point::global_coordinates<dimension_tag> corner_point;
      corner_index.iy += fgll;
      specfem::assembly::load_on_device(corner_index, mesh, corner_point);
      type_real dist = specfem::point::distance(current_point, corner_point);
      distance_acc.update(dist);
      element_distance_acc.update(point_index.ispec, dist);
    }
  }

  // Z direction
  if (point_index.iz == 0) {
    specfem::point::index<dimension_tag> corner_index = point_index;
    specfem::point::global_coordinates<dimension_tag> corner_point;
    corner_index.iz += fgll;
    specfem::assembly::load_on_device(corner_index, mesh, corner_point);
    type_real dist = specfem::point::distance(current_point, corner_point);
    distance_acc.update(dist);
    element_distance_acc.update(point_index.ispec, dist);
  }
}

} // namespace specfem::assembly::info::impl