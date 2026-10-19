#include "specfem/assembly/mesh/dim3/impl/free_surface_geometry.hpp"

#include "specfem/assembly/mesh.hpp"
#include "specfem/mesh.hpp"

// Regional mesh: the Z_MAX acoustic free surface, remapped to compute-element
// ordering.
specfem::assembly::mesh_impl::free_surface_geometry::free_surface_geometry(
    const specfem::mesh::cartesian3d_mesh &mesh,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh) {
  const auto &surface = mesh.boundaries.acoustic_free_surface;
  nfaces = surface.nelem_acoustic_surface;
  element_index = Kokkos::View<int *, Kokkos::HostSpace>(
      "specfem::assembly::mesh::free_surface::element_index", nfaces);
  face_type =
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>(
          "specfem::assembly::mesh::free_surface::face_type", nfaces);
  const auto &mesh_to_compute = assembly_mesh.h_mesh_to_compute;
  for (int f = 0; f < nfaces; ++f) {
    element_index(f) = mesh_to_compute(surface.index_mapping(f));
    face_type(f) = surface.type(f);
  }
}

// Globe mesh: the free surface is kept as geometry (globe.free_surface), not as
// a boundary condition; element indices are zero-based mesh indices.
specfem::assembly::mesh_impl::free_surface_geometry::free_surface_geometry(
    const specfem::mesh::globe3d_mesh &mesh,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh) {
  const auto &surface = mesh.globe.free_surface;
  nfaces = static_cast<int>(surface.elements.size());
  element_index = Kokkos::View<int *, Kokkos::HostSpace>(
      "specfem::assembly::mesh::free_surface::element_index", nfaces);
  face_type =
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>(
          "specfem::assembly::mesh::free_surface::face_type", nfaces);
  const auto &mesh_to_compute = assembly_mesh.h_mesh_to_compute;
  for (int f = 0; f < nfaces; ++f) {
    element_index(f) = mesh_to_compute(surface.elements[f]);
    face_type(f) = surface.faces[f];
  }
}
