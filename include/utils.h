#ifndef UTILS_H
#define UTILS_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"
#include "../include/specfem_setup.hpp"
#include <tuple>

namespace specfem {
namespace utilities {
struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

struct return_holder {
  type_real rho, mu, kappa, qmu, qkappa, lambdaplus2mu;
};

std::tuple<type_real, type_real, int, int>
locate(const specfem::kokkos::HostView2d<type_real> coord,
       const specfem::kokkos::HostMirror3d<int> ibool,
       const specfem::kokkos::HostMirror1d<type_real> xigll,
       const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
       const type_real x_source, const type_real z_source,
       const specfem::kokkos::HostView2d<type_real> coorg,
       const specfem::kokkos::HostView2d<int> knods, const int npgeo,
       const specfem::MPI::MPI *mpi);

void check_locations(const type_real x, const type_real z, const type_real xmin,
                     const type_real xmax, const type_real zmin,
                     const type_real zmax, const specfem::MPI::MPI *mpi);

int compute_nglob(const specfem::kokkos::HostMirror3d<int> ibool);
} // namespace utilities
} // namespace specfem

#endif
