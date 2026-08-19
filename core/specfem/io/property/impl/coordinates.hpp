#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace io {
namespace property_impl {

/**
 * @brief Write the GLL coordinates of @p element_indices as "X"/"Z" datasets
 *        of @p group.
 *
 * Coordinates are staged through plain row-major host views so the file
 * payload is independent of the build's SIMD tiling (see sub_block.hpp).
 *
 * @tparam GroupType Output library group type
 * @tparam ElementIndicesType Index range of the group's elements
 * @param group File group receiving the coordinate datasets
 * @param mesh Assembly mesh providing the host coordinate view
 * @param element_indices Global element indices of the group
 */
template <typename GroupType, typename ElementIndicesType>
void write_coordinates(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const ElementIndicesType &element_indices) {

  const int ngllz = mesh.element_grid.ngllz;
  const int ngllx = mesh.element_grid.ngllx;
  const int n_elements = element_indices.size();

  using CoordView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  CoordView x("xcoordinates", n_elements, ngllz, ngllx);
  CoordView z("zcoordinates", n_elements, ngllz, ngllx);
  for (int i = 0; i < n_elements; i++) {
    const int ispec = element_indices(i);
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        x(i, iz, ix) = mesh.h_coord(0, ispec, iz, ix);
        z(i, iz, ix) = mesh.h_coord(1, ispec, iz, ix);
      }
    }
  }
  group.createDataset("X", x).write();
  group.createDataset("Z", z).write();
}

/**
 * @brief Write the GLL coordinates of @p element_indices as "X"/"Y"/"Z"
 *        datasets of @p group.
 *
 * Coordinates are staged through plain row-major host views so the file
 * payload is independent of the build's SIMD tiling (see sub_block.hpp).
 *
 * @tparam GroupType Output library group type
 * @tparam ElementIndicesType Index range of the group's elements
 * @param group File group receiving the coordinate datasets
 * @param mesh Assembly mesh providing the host coordinate view
 * @param element_indices Global element indices of the group
 */
template <typename GroupType, typename ElementIndicesType>
void write_coordinates(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const ElementIndicesType &element_indices) {

  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;
  const int n_elements = element_indices.size();

  using CoordView =
      Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  CoordView x("xcoordinates", n_elements, ngllz, nglly, ngllx);
  CoordView y("ycoordinates", n_elements, ngllz, nglly, ngllx);
  CoordView z("zcoordinates", n_elements, ngllz, nglly, ngllx);
  for (int i = 0; i < n_elements; i++) {
    const int ispec = element_indices(i);
    for (int iz = 0; iz < ngllz; iz++) {
      for (int iy = 0; iy < nglly; iy++) {
        for (int ix = 0; ix < ngllx; ix++) {
          x(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 0);
          y(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 1);
          z(i, iz, iy, ix) = mesh.h_coord(ispec, iz, iy, ix, 2);
        }
      }
    }
  }
  group.createDataset("X", x).write();
  group.createDataset("Y", y).write();
  group.createDataset("Z", z).write();
}

} // namespace property_impl
} // namespace io
} // namespace specfem
