#include "IO/mesh/impl/fortran/read_elements.hpp"
#include "IO/fortranio/interface.hpp"
#include "enumerations/interface.hpp"
#include "mesh/elements/axial_elements.hpp"
#include "mesh/elements/tangential_elements.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::elements::axial_elements<specfem::dimension::type::dim2>
specfem::IO::mesh::impl::fortran::read_axial_elements(
    std::ifstream &stream, const int nelem_on_the_axis, const int nspec,
    const specfem::MPI::MPI *mpi) {

  int ispec;

  specfem::mesh::elements::axial_elements<specfem::dimension::type::dim2>
      axial_elements(nspec);

  for (int inum = 0; inum < nelem_on_the_axis; inum++) {
    specfem::IO::fortran_read_line(stream, &ispec);
    if (ispec < 0 || ispec > nspec - 1)
      throw std::runtime_error(
          "ispec out of range when reading axial elements");
    axial_elements.is_on_the_axis(ispec) = true;
  }

  return axial_elements;
}

specfem::mesh::elements::tangential_elements<specfem::dimension::type::dim2>
specfem::IO::mesh::impl::fortran::read_tangential_elements(
    std::ifstream &stream, const int nnodes_tangential_curve) {
  type_real xread, yread;

  auto tangential_elements = specfem::mesh::elements::tangential_elements<
      specfem::dimension::type::dim2>(nnodes_tangential_curve);

  specfem::IO::fortran_read_line(stream,
                                 &tangential_elements.force_normal_to_surface,
                                 &tangential_elements.rec_normal_to_surface);

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      specfem::IO::fortran_read_line(stream, &xread, &yread);
      tangential_elements.x(inum) = xread;
      tangential_elements.y(inum) = yread;
    }
  } else {
    tangential_elements.force_normal_to_surface = false;
    tangential_elements.rec_normal_to_surface = false;
  }

  return tangential_elements;
}
