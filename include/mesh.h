#ifndef MESH_H
#define MESH_H

#include "../include/boundaries.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/elements.h"
#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/mpi_interfaces.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"
#include "../include/surfaces.h"
#include <Kokkos_Core.hpp>

namespace specfem {

struct properties {
  int numat, ngnod, nspec, pointsdisp, nelemabs, nelem_acforcing,
      nelem_acoustic_surface, num_fluid_solid_edges, num_fluid_poro_edges,
      num_solid_poro_edges, nnodes_tangential_curve, nelem_on_the_axis;
  bool plot_lowerleft_corner_only;
};

namespace materials {
struct material_ind {
  specfem::HostView1d<int> region_CPML, kmato;
  specfem::HostView2d<int> knods;

  material_ind(){};
  material_ind(const int nspec, const int ngnod);
};
} // namespace materials

struct mesh {
  int npgeo, nspec, nproc;
  specfem::HostView2d<type_real> coorg;
  specfem::materials::material_ind material_ind;
  specfem::interfaces::interface interface;
  specfem::boundaries::absorbing_boundary abs_boundary;
  specfem::properties parameters;
  specfem::surfaces::acoustic_free_surface acfree_surface;
  specfem::boundaries::forcing_boundary acforcing_boundary;
  specfem::elements::tangential_elements tangential_nodes;
  specfem::elements::axial_elements axial_nodes;
  specfem::compute::compute compute;

  // mesh(const std::string filename);

  void setup(std::vector<specfem::material *> &materials,
             const quadrature::quadrature &quadx,
             const quadrature::quadrature &quadz, specfem::MPI *mpi);
};
} // namespace specfem

#endif
