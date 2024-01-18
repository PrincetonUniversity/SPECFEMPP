#include "mesh/material_indic/material_indic.hpp"
#include "fortranio/interface.hpp"
#include "kokkos_abstractions.h"
#include <vector>

specfem::mesh::material_ind::material_ind(const int nspec, const int ngnod) {
  this->region_CPML =
      specfem::kokkos::HostView1d<int>("specfem::mesh::region_CPML", nspec);
  this->kmato =
      specfem::kokkos::HostView1d<int>("specfem::mesh::region_CPML", nspec);

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->kmato(ispec) = -1;
  }
  return;
}

specfem::mesh::material_ind::material_ind(
    std::ifstream &stream, const int ngnod, const int nspec, const int numat,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::MPI::MPI *mpi) {
  std::vector<int> knods_read(ngnod, -1);
  int n, kmato_read, pml_read;

  // Allocate views
  *this = specfem::mesh::material_ind(nspec, ngnod);

  // Read an assign material values, coordinate numbering, PML association
  for (int ispec = 0; ispec < nspec; ispec++) {
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    specfem::fortran_IO::fortran_read_line(stream, &n, &kmato_read, &knods_read,
                                           &pml_read);

    // material association
    if (n < 1 || n > nspec) {
      throw std::runtime_error("Error reading mato properties");
    }
    this->kmato(n - 1) = kmato_read - 1;
    this->region_CPML(n - 1) = pml_read;

    // element control node indices (ipgeo)
    for (int i = 0; i < ngnod; i++) {
      if (knods_read[i] == 0)
        throw std::runtime_error("Error reading knods (node_id) values");

      knods(i, n - 1) = knods_read[i] - 1;
    }
  }

  for (int ispec = 0; ispec < nspec; ispec++) {
    int imat = this->kmato(ispec);
    if (imat < 0 || imat >= numat) {
      throw std::runtime_error(
          "Error reading material properties. Invalid material ID number");
    }
  }
}
