#ifndef _ACOUSTIC_FREE_SURFACE_HPP
#define _ACOUSTIC_FREE_SURFACE_HPP

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace boundaries {

struct acoustic_free_surface {
  specfem::kokkos::HostView1d<int> numacfree_surface, typeacfree_surface, e1,
      e2, ixmin, ixmax, izmin, izmax;

  acoustic_free_surface(){};
  acoustic_free_surface(const int nelem_acoustic_surface);
  acoustic_free_surface(std::ifstream &stream, const int nelem_acoustic_surface,
                        const specfem::MPI::MPI *mpi);
};
} // namespace boundaries
} // namespace mesh
} // namespace specfem

#endif
