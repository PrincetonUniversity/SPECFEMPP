#include "injection_surface.hpp"
#include <Kokkos_Core.hpp>

specfem::mesh::InjectionSurface<specfem::element::dimension_tag::dim3>::
    InjectionSurface(const InjectionSurfaceKind kind,
                     const std::array<bool, nsides> &sides, const int nfaces)
    : kind(kind), sides(sides), nfaces(nfaces) {
  if (nfaces > 0) {
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::InjectionSurface::index_mapping", nfaces);
    this->type =
        Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>(
            "specfem::mesh::InjectionSurface::type", nfaces);
    this->side = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::InjectionSurface::side", nfaces);
  }
}
