#ifndef ELEMENTS_H
#define ELEMENTS_H

#include "../include/kokkos_abstractions.h"

namespace specfem {
namespace elements {

struct tangential_elements {
  bool force_normal_to_surface, rec_normal_to_surface;
  specfem::HostView1d<type_real> x, y;
  tangential_elements(){};
  tangential_elements(const int nnodes_tangential_curve);
};

struct axial_elements {
  specfem::HostView1d<bool> is_on_the_axis;
  axial_elements(){};
  axial_elements(const int nspec);
};

} // namespace elements
} // namespace specfem

#endif
