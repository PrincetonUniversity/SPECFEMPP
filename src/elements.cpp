#include "../include/elements.h"

specfem::elements::tangential_elements::tangential_elements(
    const int nnodes_tangential_curve) {
  if (nnodes_tangential_curve > 0) {
    this->x = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::x", nnodes_tangential_curve);
    this->y = specfem::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::y", nnodes_tangential_curve);
  } else {
    this->x =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::x", 1);
    this->y =
        specfem::HostView1d<type_real>("specfem::mesh::tangential_nodes::y", 1);
  }

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      this->x(inum) = 0.0;
      this->y(inum) = 0.0;
    }
  } else {
    this->x(1) = 0.0;
    this->y(1) = 0.0;
  }
  return;
}

specfem::elements::axial_elements::axial_elements(const int nspec) {
  this->is_on_the_axis = specfem::HostView1d<bool>(
      "specfem::mesh::axial_element::is_on_the_axis", nspec);

  for (int inum = 0; inum < nspec; inum++) {
    this->is_on_the_axis(nspec) = false;
  }

  return;
}
