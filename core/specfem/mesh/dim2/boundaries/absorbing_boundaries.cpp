#include "absorbing_boundary.hpp"
#include "specfem/io.hpp"

#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::absorbing_boundary<specfem::element::dimension_tag::dim2>::
    absorbing_boundary(const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    this->nelements = num_abs_boundary_faces;
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::absorbing_boundary::index_mapping",
        num_abs_boundary_faces);
    this->type =
        Kokkos::View<specfem::mesh_entity::dim2::type *, Kokkos::HostSpace>(
            "specfem::mesh::absorbing_boundary::type", num_abs_boundary_faces);
  } else {
    this->nelements = 0;
  }

  return;
}
