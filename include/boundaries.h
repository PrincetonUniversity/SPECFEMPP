#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace boundaries {

struct absorbing_boundary {
  specfem::HostView1d<int> numabs, abs_boundary_type, ibegin_edge1,
      ibegin_edge2, ibegin_edge3, ibegin_edge4, iend_edge1, iend_edge2,
      iend_edge3, iend_edge4, ib_bottom, ib_top, ib_right, ib_left;
  specfem::HostView2d<bool> codeabs, codeabscorner;
  absorbing_boundary(){};
  absorbing_boundary(const int num_abs_boundaries_faces);
  absorbing_boundary(std::ifstream &stream, int num_abs_boundary_faces,
                     const int nspec, const specfem::MPI *mpi);
};

struct forcing_boundary {
  specfem::HostView1d<int> numacforcing, typeacforcing, ibegin_edge1,
      ibegin_edge2, ibegin_edge3, ibegin_edge4, iend_edge1, iend_edge2,
      iend_edge3, iend_edge4, ib_bottom, ib_top, ib_right, ib_left;
  specfem::HostView2d<bool> codeacforcing;
  forcing_boundary(){};
  forcing_boundary(const int nelement_acforcing);
  forcing_boundary(std::ifstream &stream, const int nelement_acforcing,
                   const int nspec, const specfem::MPI *mpi);
};

} // namespace boundaries
} // namespace specfem

#endif
