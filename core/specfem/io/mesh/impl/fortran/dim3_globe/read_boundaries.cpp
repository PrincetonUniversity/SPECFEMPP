#include "specfem/io/mesh/impl/fortran/dim3_globe/read_boundaries.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

specfem::mesh::globe_boundary_surface
specfem::io::mesh::impl::fortran::dim3_globe::read_surface(
    std::ifstream &stream, const int nspec) {
  specfem::mesh::globe_boundary_surface result;
  int nfaces = 0;
  specfem::io::fortran_read_line(stream, &nfaces);
  if (nfaces < 0) {
    throw std::runtime_error("Negative face count in globe mesh database");
  }

  result.elements.resize(nfaces);
  std::vector<int> faces(nfaces);
  if (nfaces > 0) {
    specfem::io::fortran_read_line(stream, &result.elements, &faces);
  }

  result.faces.resize(nfaces);
  for (int iface = 0; iface < nfaces; ++iface) {
    if (result.elements[iface] < 1 || result.elements[iface] > nspec) {
      throw std::runtime_error(
          "Invalid boundary element in globe mesh database");
    }
    --result.elements[iface];
    result.faces[iface] =
        specfem::io::mesh::impl::fortran::dim3_globe::to_face(faces[iface]);
  }

  return result;
}

void specfem::io::mesh::impl::fortran::dim3_globe::read_boundaries(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh) {
  using Dimension = specfem::element::dimension_tag;

  auto &globe = mesh.globe;
  globe.free_surface = read_surface(stream, mesh.nspec);
  globe.cmb = read_surface(stream, mesh.nspec);
  globe.icb = read_surface(stream, mesh.nspec);
  globe.ocean_load = read_surface(stream, mesh.nspec);

  specfem::mesh::absorbing_boundary<Dimension::dim3> absorbing(0);
  specfem::mesh::acoustic_free_surface<Dimension::dim3> free_surface(
      static_cast<int>(globe.free_surface.elements.size()));

  Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      free_surface_elements_view(globe.free_surface.elements.data(),
                                 globe.free_surface.elements.size());
  Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::LayoutLeft,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      free_surface_faces_view(globe.free_surface.faces.data(),
                              globe.free_surface.faces.size());
  auto free_surface_index_mapping = free_surface.index_mapping;
  auto free_surface_type = free_surface.type;
  Kokkos::parallel_for(
      "specfem::io::mesh::dim3_globe::read_boundaries::free_surface",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
          0, static_cast<int>(globe.free_surface.elements.size())),
      [=](const int iface) {
        free_surface_index_mapping(iface) = free_surface_elements_view(iface);
        free_surface_type(iface) = free_surface_faces_view(iface);
      });
  Kokkos::fence();

  mesh.boundaries = { absorbing, free_surface };
}
