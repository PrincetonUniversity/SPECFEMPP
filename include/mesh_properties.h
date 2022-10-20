#ifndef MESH_PROPERTIES_H
#define MESH_PROPERTIES_H

#include "../include/specfem_mpi.h"

namespace specfem {
struct properties {
  int numat, ngnod, nspec, pointsdisp, nelemabs, nelem_acforcing,
      nelem_acoustic_surface, num_fluid_solid_edges, num_fluid_poro_edges,
      num_solid_poro_edges, nnodes_tangential_curve, nelem_on_the_axis;
  bool plot_lowerleft_corner_only;

  properties(){};
  properties(std::ifstream &stream, const specfem::MPI *mpi);
};
} // namespace specfem

#endif
