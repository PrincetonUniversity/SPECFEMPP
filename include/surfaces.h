#ifndef SURFACES_H
#define SURFACES_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {

/**
 * Define elements on a surface
 *
 * @note Need to still document this section
 *
 */
namespace surfaces {

struct acoustic_free_surface {
  specfem::kokkos::HostView1d<int> numacfree_surface, typeacfree_surface, e1,
      e2, ixmin, ixmax, izmin, izmax;

  acoustic_free_surface(){};
  acoustic_free_surface(const int nelem_acoustic_surface);
  acoustic_free_surface(std::ifstream &stream, const int nelem_acoustic_surface,
                        const specfem::MPI::MPI *mpi);
};

} // namespace surfaces
} // namespace specfem

#endif
