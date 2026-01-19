#include "acoustic_free_surface.hpp"
#include "io/fortranio/interface.hpp"

#include <Kokkos_Core.hpp>

specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2>::
    acoustic_free_surface(const int nelem_acoustic_surface)
    : nelem_acoustic_surface(nelem_acoustic_surface) {
  if (nelem_acoustic_surface > 0) {
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::acoustic_free_surface::index_mapping",
        nelem_acoustic_surface);
    this->type =
        Kokkos::View<specfem::mesh_entity::dim2::type *, Kokkos::HostSpace>(
            "specfem::mesh::acoustic_free_surface::type",
            nelem_acoustic_surface);
  }
  return;
}
