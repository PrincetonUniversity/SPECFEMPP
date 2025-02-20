#include "IO/fortranio/interface.hpp"
#include "enumerations/interface.hpp"
#include "mesh/dim2/elements/elements.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::elements::axial_elements<
    specfem::dimension::type::dim2>::axial_elements(const int nspec) {
  this->is_on_the_axis = specfem::kokkos::HostView1d<bool>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    this->is_on_the_axis(inum) = false;
  }

  return;
}
