#include "mapping.hpp"
#include "specfem/parallel_configuration.hpp"
#include <algorithm>
#include <cmath>

namespace specfem::test {
namespace mesh_utilities {

//-------------------------- 2D Function Implementations
//------------------------//

type_real compute_spatial_tolerance(const std::vector<point_2d> &points,
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

std::vector<point_2d>
flatten_coordinates(const HostView4d &global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxz = ngll * ngll;

  std::vector<point_2d> points(nspec * ngllxz);

  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;

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

void sort_points_spatially(std::vector<point_2d> &points) {
  std::sort(points.begin(), points.end(),
            [&](const point_2d &p1, const point_2d &p2) {
              if (p1.x != p2.x) {
                return p1.x < p2.x;
              }
              return p1.z < p2.z;
            });
}

int assign_global_numbering(std::vector<point_2d> &points,
                            type_real tolerance) {
  if (points.empty())
    return 0;

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

std::vector<point_2d>
reorder_to_original_layout(const std::vector<point_2d> &sorted_points) {
  std::vector<point_2d> reordered(sorted_points.size());

  for (int i = 0; i < sorted_points.size(); i++) {
    int iloc = sorted_points[i].iloc;
    reordered[iloc] = sorted_points[i];
  }

  return reordered;
}

bounding_box_2d compute_bounding_box(const std::vector<point_2d> &points) {
  bounding_box_2d bbox;

  for (const auto &p : points) {
    bbox.xmin = std::min(bbox.xmin, p.x);
    bbox.xmax = std::max(bbox.xmax, p.x);
    bbox.zmin = std::min(bbox.zmin, p.z);
    bbox.zmax = std::max(bbox.zmax, p.z);
  }

  return bbox;
}

std::tuple<Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point_2d> &reordered_points,
                         int nspec, int ngll, int nglob) {

  // Create coordinate arrays (host-based since assign_numbering is host-only)
  auto index_mapping =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "index_mapping", nspec, ngll, ngll);
  auto coord =
      Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>(
          "coord", 2, nspec, ngll, ngll);

  std::vector<int> iglob_counted(nglob, -1);
  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;
  int iloc = 0;
  int inum = 0;

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int ix = 0; ix < ngll; ix++) {
        for (int ielement = 0; ielement < chunk_size; ielement++) {
          int ispec = ichunk + ielement;
          if (ispec >= nspec)
            break;
          if (iglob_counted[reordered_points[iloc].iglob] == -1) {
            const type_real x_cor = reordered_points[iloc].x;
            const type_real z_cor = reordered_points[iloc].z;

            iglob_counted[reordered_points[iloc].iglob] = inum;
            index_mapping(ispec, iz, ix) = inum;
            coord(0, ispec, iz, ix) = x_cor;
            coord(1, ispec, iz, ix) = z_cor;
            inum++;
          } else {
            index_mapping(ispec, iz, ix) =
                iglob_counted[reordered_points[iloc].iglob];
            coord(0, ispec, iz, ix) = reordered_points[iloc].x;
            coord(1, ispec, iz, ix) = reordered_points[iloc].z;
          }
          iloc++;
        }
      }
    }
  }

  int ngllxz = ngll * ngll;

  // assembly should reduce number of unique global nodes if there are shared
  // nodes. nspec = 1 means no sharing, so nglob should equal nspec*ngllxz
  if (nspec > 1) {
    assert(nglob < (nspec * ngllxz));
  } else {
    assert(nglob == (nspec * ngllxz));
  }
  assert(inum == nglob);

  return std::make_tuple(index_mapping, coord, inum);
}

//-------------------------- 3D Function Implementations
//----------------------//

type_real compute_spatial_tolerance(const std::vector<point_3d> &points,
                                    int nspec, int ngllxyz) {
  type_real xtypdist = std::numeric_limits<type_real>::max();

  for (int ispec = 0; ispec < nspec; ispec++) {
    type_real xmax = std::numeric_limits<type_real>::min();
    type_real xmin = std::numeric_limits<type_real>::max();
    type_real ymax = std::numeric_limits<type_real>::min();
    type_real ymin = std::numeric_limits<type_real>::max();
    type_real zmax = std::numeric_limits<type_real>::min();
    type_real zmin = std::numeric_limits<type_real>::max();

    for (int xyz = 0; xyz < ngllxyz; xyz++) {
      int iloc = ispec * ngllxyz + xyz;
      xmax = std::max(xmax, points[iloc].x);
      xmin = std::min(xmin, points[iloc].x);
      ymax = std::max(ymax, points[iloc].y);
      ymin = std::min(ymin, points[iloc].y);
      zmax = std::max(zmax, points[iloc].z);
      zmin = std::min(zmin, points[iloc].z);
    }

    xtypdist = std::min(xtypdist, xmax - xmin);
    xtypdist = std::min(xtypdist, ymax - ymin);
    xtypdist = std::min(xtypdist, zmax - zmin);
  }

  return 1e-6 * xtypdist;
}

std::vector<point_3d>
flatten_coordinates(const HostView5d &global_coordinates) {

  int nspec = global_coordinates.extent(0);
  int ngll = global_coordinates.extent(1);
  int ngllxyz = ngll * ngll * ngll; // 3D: ngll^3 points per element

  std::vector<point_3d> points(nspec * ngllxyz);

  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;

  int iloc = 0;
  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int iy = 0; iy < ngll; iy++) {
        for (int ix = 0; ix < ngll; ix++) {
          for (int ielement = 0; ielement < chunk_size; ielement++) {
            int ispec = ichunk + ielement;
            if (ispec >= nspec)
              break;
            // For 3D: access pattern (ispec, iz, iy, ix, icoord) with 5D array
            points[iloc].x = global_coordinates(ispec, iz, iy, ix, 0); // x
            points[iloc].y = global_coordinates(ispec, iz, iy, ix, 1); // y
            points[iloc].z = global_coordinates(ispec, iz, iy, ix, 2); // z
            points[iloc].iloc = iloc;
            iloc++;
          }
        }
      }
    }
  }

  return points;
}

void sort_points_spatially(std::vector<point_3d> &points) {
  std::sort(points.begin(), points.end(),
            [&](const point_3d &p1, const point_3d &p2) {
              if (p1.x != p2.x) {
                return p1.x < p2.x;
              }
              if (p1.y != p2.y) {
                return p1.y < p2.y;
              }
              return p1.z < p2.z;
            });
}

int assign_global_numbering(std::vector<point_3d> &points,
                            type_real tolerance) {
  if (points.empty())
    return 0;

  int ig = 0;
  points[0].iglob = ig;

  for (int i = 1; i < points.size(); i++) {
    if ((std::abs(points[i].x - points[i - 1].x) > tolerance) ||
        (std::abs(points[i].y - points[i - 1].y) > tolerance) ||
        (std::abs(points[i].z - points[i - 1].z) > tolerance)) {
      ig++;
    }
    points[i].iglob = ig;
  }

  return ig + 1;
}

std::vector<point_3d>
reorder_to_original_layout(const std::vector<point_3d> &sorted_points) {
  std::vector<point_3d> reordered(sorted_points.size());

  for (int i = 0; i < sorted_points.size(); i++) {
    int iloc = sorted_points[i].iloc;
    reordered[iloc] = sorted_points[i];
  }

  return reordered;
}

bounding_box_3d compute_bounding_box(const std::vector<point_3d> &points) {
  bounding_box_3d bbox;

  for (const auto &p : points) {
    bbox.xmin = std::min(bbox.xmin, p.x);
    bbox.xmax = std::max(bbox.xmax, p.x);
    bbox.ymin = std::min(bbox.ymin, p.y);
    bbox.ymax = std::max(bbox.ymax, p.y);
    bbox.zmin = std::min(bbox.zmin, p.z);
    bbox.zmax = std::max(bbox.zmax, p.z);
  }

  return bbox;
}

std::tuple<Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           int>
create_coordinate_arrays(const std::vector<point_3d> &reordered_points,
                         int nspec, int ngll, int nglob) {

  // Create coordinate arrays for 3D
  auto index_mapping =
      Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "index_mapping", nspec, ngll, ngll, ngll); // (ispec, iz, iy, ix)
  auto coord =
      Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>(
          "coord", nspec, ngll, ngll, ngll, 3); // (ispec, iz, iy, ix, icoord)

  std::vector<int> iglob_counted(nglob, -1);
  constexpr int chunk_size =
      specfem::parallel_configuration::storage_chunk_size;
  int iloc = 0;
  int inum = 0;

  for (int ichunk = 0; ichunk < nspec; ichunk += chunk_size) {
    for (int iz = 0; iz < ngll; iz++) {
      for (int iy = 0; iy < ngll; iy++) {
        for (int ix = 0; ix < ngll; ix++) {
          for (int ielement = 0; ielement < chunk_size; ielement++) {
            int ispec = ichunk + ielement;
            if (ispec >= nspec)
              break;
            if (iglob_counted[reordered_points[iloc].iglob] == -1) {
              const type_real x_cor = reordered_points[iloc].x;
              const type_real y_cor = reordered_points[iloc].y;
              const type_real z_cor = reordered_points[iloc].z;

              iglob_counted[reordered_points[iloc].iglob] = inum;
              index_mapping(ispec, iz, iy, ix) = inum;
              coord(ispec, iz, iy, ix, 0) = x_cor;
              coord(ispec, iz, iy, ix, 1) = y_cor;
              coord(ispec, iz, iy, ix, 2) = z_cor;
              inum++;
            } else {
              index_mapping(ispec, iz, iy, ix) =
                  iglob_counted[reordered_points[iloc].iglob];
              coord(ispec, iz, iy, ix, 0) = reordered_points[iloc].x;
              coord(ispec, iz, iy, ix, 1) = reordered_points[iloc].y;
              coord(ispec, iz, iy, ix, 2) = reordered_points[iloc].z;
            }
            iloc++;
          }
        }
      }
    }
  }

  int ngllxyz = ngll * ngll * ngll;
  assert(nglob != (nspec * ngllxyz));
  assert(inum == nglob);

  return std::make_tuple(index_mapping, coord, inum);
}

} // namespace mesh_utilities
} // namespace specfem::test
