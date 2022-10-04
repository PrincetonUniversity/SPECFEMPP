#ifndef SURFACES_H
#define SURFACES_H

#include "../include/kokkos_abstractions.h"

namespace specfem {
namespace surfaces {

struct acoustic_free_surface {
  specfem::HostView1d<int> numacfree_surface, typeacfree_surface, e1, e2, ixmin,
      ixmax, izmin, izmax;

  acoustic_free_surface(){};
  acoustic_free_surface(const int nelem_acoustic_surface);
};

} // namespace surfaces
} // namespace specfem

#endif
