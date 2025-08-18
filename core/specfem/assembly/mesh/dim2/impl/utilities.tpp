#pragma once

#include "utilities.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include <algorithm>
#include <cmath>

namespace specfem {
namespace assembly {
namespace mesh_impl {
namespace dim2 {
namespace utilities {

type_real compute_spatial_tolerance(const std::vector<point>& points,
                                   int nspec, int ngllxz) {
  type_real xtypdist = std::numeric_limits<type_real>::max();

  for (int ispec = 0; ispec < nspec; ispec++) {
    type_real xmax = std::numeric_limits<type_real>::min();
    type_real xmin = std::numeric_limits<type_real>::max();
    type_real ymax = std::numeric_limits<type_real>::min();
    type_real ymin = std::numeric_limits<type_real>::max();

    for (int xz = 0; xz < ngllxz; xz++) {
      int iloc = ispec * ngllxz + xz;
      xmax = std::max(xmax, points[iloc].x);
      xmin = std::min(xmin, points[iloc].x);
      ymax = std::max(ymax, points[iloc].z);
      ymin = std::min(ymin, points[iloc].z);
    }

    xtypdist = std::min(xtypdist, xmax - xmin);
    xtypdist = std::min(xtypdist, ymax - ymin);
  }

  return 1e-6 * xtypdist;
}

std::vector<point> flatten_coordinates(
    const specfem::kokkos::HostView4d<double>& global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  std::vector<point> points(nspec * ngllxz);

  constexpr int chunk_size = specfem::parallel_config::storage_chunk_size;

  int iloc = 0;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          points[iloc].x = global_coordinates(ispec, iz, ix, 0);
          points[iloc].z = global_coordinates(ispec, iz, ix, 1);
          points[iloc].iloc = iloc;
          iloc++;
        }
      }
    }
  }

  return points;
}

void sort_points_spatially(std::vector<point>& points) {
  std::sort(points.begin(), points.end(),
            [&](const point& p1, const point& p2) {
              if (p1.x != p2.x) {
                return p1.x < p2.x;
              }
              return p1.z < p2.z;
            });
}

int assign_global_numbering(std::vector<point>& points, type_real tolerance) {
  if (points.empty()) return 0;

  int ig = 0;
  points[0].iglob = ig;

  for (int i = 1; i < points.size(); i++) {
    if ((std::abs(points[i].x - points[i - 1].x) > tolerance) ||
        (std::abs(points[i].z - points[i - 1].z) > tolerance)) {
      ig++;
    }
    points[i].iglob = ig;
  }

  return ig + 1;
}

std::vector<point> reorder_to_original_layout(const std::vector<point>& sorted_points) {
  std::vector<point> reordered(sorted_points.size());

  for (int i = 0; i < sorted_points.size(); i++) {
    int iloc = sorted_points[i].iloc;
    reordered[iloc] = sorted_points[i];
  }

  return reordered;
}

bounding_box compute_bounding_box(const std::vector<point>& points) {
  bounding_box bbox;

  for (const auto& p : points) {
    bbox.xmin = std::min(bbox.xmin, p.x);
    bbox.xmax = std::max(bbox.xmax, p.x);
    bbox.zmin = std::min(bbox.zmin, p.z);
    bbox.zmax = std::max(bbox.zmax, p.z);
  }

  return bbox;
}

} // namespace utilities
} // namespace dim2
} // namespace mesh_impl
} // namespace assembly
} // namespace specfem
