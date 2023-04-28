#include "fortranio/interface.hpp"
#include "mesh/elements/elements.hpp"
#include "specfem_mpi.h"

specfem::mesh::elements::axial_elements::axial_elements(const int nspec) {
  this->is_on_the_axis = specfem::kokkos::HostView1d<bool>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    this->is_on_the_axis(nspec) = false;
  }

  return;
}

specfem::mesh::elements::axial_elements::axial_elements(
    std::ifstream &stream, const int nelem_on_the_axis, const int nspec,
    const specfem::MPI::MPI *mpi) {
  int ispec;

  *this = specfem::mesh::elements::axial_elements(nspec);
  for (int inum = 0; inum < nelem_on_the_axis; inum++) {
    specfem::fortran_IO::fortran_read_line(stream, &ispec);
    if (ispec < 0 || ispec > nspec - 1)
      throw std::runtime_error(
          "ispec out of range when reading axial elements");
    this->is_on_the_axis(ispec) = true;
  }

  return;
}
