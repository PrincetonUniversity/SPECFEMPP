#include "specfem/io/mesh/impl/fortran/dim3_globe/read_boundaries.hpp"

#include "specfem/io.hpp"
#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"

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

  // The free surface is kept in globe.free_surface as a geometric surface
  // only -- for depth resolution, the ocean load and surface output -- and is
  // deliberately not routed into acoustic_free_surface. It is the top of the
  // elastic crust/mantle, where traction-free is the natural condition of the
  // weak form; as a boundary condition it would put a zero-pressure constraint
  // on any acoustic element that owned one of its faces. The database carries
  // no absorbing faces either.
  mesh.boundaries = { specfem::mesh::absorbing_boundary<Dimension::dim3>(0),
                      specfem::mesh::acoustic_free_surface<Dimension::dim3>(
                          0) };
}
