#include "acoustic_free_surface.hpp"
#include <Kokkos_Core.hpp>

specfem::mesh::acoustic_free_surface<
    specfem::element::dimension_tag::dim3>::acoustic_free_surface(const int n)
    : nelem_acoustic_surface(n) {
  if (n > 0) {
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::acoustic_free_surface::index_mapping", n);
    this->type =
        Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>(
            "specfem::mesh::acoustic_free_surface::type", n);
  }
}
