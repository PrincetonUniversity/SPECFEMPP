#ifndef MESH_H
#define MESH_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
struct prop {
  int numat, ngnod, nspec, pointsdisp, nelemabs, nelem_acforcing,
      nelem_acoustic_surface, num_fluid_solid_edges, num_fluid_poro_edges,
      num_solid_poro_edges, nnodes_tagential_curve, nelem_on_the_axis;
  bool plot_lowerleft_corner_only;
};

struct mesh {
  int npgeo, nspec, nproc;
  specfem::HostView2d<type_real> coorg;
  prop properties;
};
} // namespace specfem

#endif
