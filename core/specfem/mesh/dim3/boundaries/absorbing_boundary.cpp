#include "absorbing_boundary.hpp"
#include <Kokkos_Core.hpp>

specfem::mesh::absorbing_boundary<specfem::element::dimension_tag::dim3>::
    absorbing_boundary(const int nelements)
    : nelements(nelements) {
  if (nelements > 0) {
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::absorbing_boundary::index_mapping", nelements);
    this->type =
        Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>(
            "specfem::mesh::absorbing_boundary::type", nelements);
  }
}
