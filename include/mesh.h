#ifndef MESH_H
#define MESH_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  specfem::HostView1d<int> my_neighbors, my_nelmnts_neighbors;
  specfem::HostView3d<int> my_interfaces;
};

struct prop {
  int numat, ngnod, nspec, pointsdisp, nelemabs, nelem_acforcing,
      nelem_acoustic_surface, num_fluid_solid_edges, num_fluid_poro_edges,
      num_solid_poro_edges, nnodes_tangential_curve, nelem_on_the_axis;
  bool plot_lowerleft_corner_only;
};

struct absorbing_boundary {
  specfem::HostView1d<int> numabs, abs_boundary_type, ibegin_edge1,
      ibegin_edge2, ibegin_edge3, ibegin_edge4, iend_edge1, iend_edge2,
      iend_edge3, iend_edge4, ib_bottom, ib_top, ib_right, ib_left;
  specfem::HostView2d<bool> codeabs, codeabscorner;
};

struct forcing_boundary {
  specfem::HostView1d<int> numacforcing, typeacforcing, ibegin_edge1,
      ibegin_edge2, ibegin_edge3, ibegin_edge4, iend_edge1, iend_edge2,
      iend_edge3, iend_edge4, ib_bottom, ib_top, ib_right, ib_left;
  specfem::HostView2d<bool> codeacforcing;
};

struct acoustic_free_surface {
  specfem::HostView1d<int> numacfree_surface, typeacfree_surface, e1, e2, ixmin,
      ixmax, izmin, izmax;
};

struct tangential_elements {
  bool force_normal_to_surface, rec_normal_to_surface;
  specfem::HostView1d<type_real> x, y;
};

struct axial_elements {
  specfem::HostView1d<bool> is_on_the_axis;
};

struct mesh {
  int npgeo, nspec, nproc;
  specfem::HostView2d<type_real> coorg;
  specfem::HostView1d<int> region_CPML, kmato;
  specfem::HostView2d<int> knods;
  interface inter;
  absorbing_boundary abs_boundary;
  prop properties;
  acoustic_free_surface acfree_surface;
  forcing_boundary acforcing_boundary;
  tangential_elements tangential_nodes;
  axial_elements axial_nodes;

  void allocate();
};
} // namespace specfem

#endif
