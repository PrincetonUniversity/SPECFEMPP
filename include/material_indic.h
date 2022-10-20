#ifndef MATERIAL_INDIC_H
#define MATERIAL_INDIC_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace materials {
struct material_ind {
  specfem::HostView1d<int> region_CPML, kmato;
  specfem::HostView2d<int> knods;

  material_ind(){};
  material_ind(const int nspec, const int ngnod);
  material_ind(std::ifstream &stream, const int ngnod, const int nspec,
               const int numat, const specfem::MPI *mpi);
};
} // namespace materials
} // namespace specfem

#endif
