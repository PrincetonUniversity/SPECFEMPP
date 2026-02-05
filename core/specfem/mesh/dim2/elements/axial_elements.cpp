#include "elements.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"

specfem::mesh::elements::axial_elements<
    specfem::element::dimension_tag::dim2>::axial_elements(const int nspec) {
  this->is_on_the_axis = Kokkos::View<bool *, Kokkos::HostSpace>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    this->is_on_the_axis(inum) = false;
  }

  return;
}
