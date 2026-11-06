#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief Linear sub-cells spanned by the GLL points of an element group.
 *
 * Each element is split into the linear cells between neighbouring GLL
 * points: `(ngllz - 1) * (ngllx - 1)` quadrilaterals in 2D and
 * `(ngllz - 1) * (nglly - 1) * (ngllx - 1)` hexahedra in 3D. Node indices
 * address the group's flattened row-major point order
 * `((e * ngllz + iz) * nglly + iy) * ngllx + ix` (2D: without `iy`), which is
 * the order of the "X"/"Y"/"Z" datasets. Nodes follow the VTK linear
 * quadrilateral order (counter-clockwise in the `(ix, iz)` plane) and the VTK
 * hexahedron order (the four nodes of the `iz` face, counter-clockwise in the
 * `(ix, iy)` plane, then the same four at `iz + 1`).
 *
 * @tparam DimensionTag Dimension of the mesh
 * @param number_of_elements Number of elements in the group
 * @param element_grid GLL layout of every element
 * @return Host view of shape `(number_of_cells, 4)` in 2D or
 *         `(number_of_cells, 8)` in 3D
 */
template <specfem::element::dimension_tag DimensionTag>
Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace>
sub_cell_connectivity(
    const int number_of_elements,
    const specfem::mesh_entity::element_grid<DimensionTag> &element_grid) {
  using ConnectivityView =
      Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace>;

  if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
    const int ngllz = element_grid.ngllz;
    const int ngllx = element_grid.ngllx;
    const std::size_t number_of_cells =
        static_cast<std::size_t>(number_of_elements) * (ngllz - 1) *
        (ngllx - 1);
    ConnectivityView connectivity("connectivity", number_of_cells, 4);

    std::size_t icell = 0;
    for (int ielement = 0; ielement < number_of_elements; ++ielement) {
      const int element_offset = ielement * ngllz * ngllx;
      for (int iz = 0; iz < ngllz - 1; ++iz) {
        for (int ix = 0; ix < ngllx - 1; ++ix) {
          const int lower_left = element_offset + iz * ngllx + ix;
          connectivity(icell, 0) = lower_left;
          connectivity(icell, 1) = lower_left + 1;
          connectivity(icell, 2) = lower_left + ngllx + 1;
          connectivity(icell, 3) = lower_left + ngllx;
          ++icell;
        }
      }
    }
    return connectivity;
  } else {
    const int ngllz = element_grid.ngllz;
    const int nglly = element_grid.nglly;
    const int ngllx = element_grid.ngllx;
    const std::size_t number_of_cells =
        static_cast<std::size_t>(number_of_elements) * (ngllz - 1) *
        (nglly - 1) * (ngllx - 1);
    ConnectivityView connectivity("connectivity", number_of_cells, 8);

    const int layer_stride = nglly * ngllx;
    std::size_t icell = 0;
    for (int ielement = 0; ielement < number_of_elements; ++ielement) {
      const int element_offset = ielement * ngllz * layer_stride;
      for (int iz = 0; iz < ngllz - 1; ++iz) {
        for (int iy = 0; iy < nglly - 1; ++iy) {
          for (int ix = 0; ix < ngllx - 1; ++ix) {
            const int corner =
                element_offset + iz * layer_stride + iy * ngllx + ix;
            const int face[4] = { corner, corner + 1, corner + ngllx + 1,
                                  corner + ngllx };
            for (int inode = 0; inode < 4; ++inode) {
              connectivity(icell, inode) = face[inode];
              connectivity(icell, inode + 4) = face[inode] + layer_stride;
            }
            ++icell;
          }
        }
      }
    }
    return connectivity;
  }
}

} // namespace impl
} // namespace io
} // namespace specfem
