#ifndef UTILS_H
#define UTILS_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"
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

// config parser routines
struct force_source {
  type_real x, z;       //< location of force source
  bool source_surf;     //< Does the source lie on the surface
                        //< In which case z value would be neglected
                        //< while setting up the source
  std::string stf_type; // type of source time function to use
  type_real f0;
  type_real angle;
  type_real vx, vz;
  type_real factor;
  type_real tshift;
};

struct moment_tensor {
  type_real x, z;       //< location of force source
  bool source_surf;     //< Does the source lie on the surface
                        //< In which case z value would be neglected
                        //< while setting up the source
  std::string stf_type; //< type of source time function to use
  type_real f0;
  type_real Mxx, Mxz, Mzz;
  type_real vx, vz;
  type_real factor;
  type_real tshift;
};

std::tuple<type_real, type_real, int, int>
locate(const specfem::HostView2d<type_real> coord,
       const specfem::HostMirror3d<int> ibool,
       const specfem::HostMirror1d<type_real> xigll,
       const specfem::HostMirror1d<type_real> zigll, const int nproc,
       const type_real x_source, const type_real z_source,
       const specfem::HostView2d<type_real> coorg,
       const specfem::HostView2d<int> knods, const int npgeo,
       const specfem::MPI::MPI *mpi);

void check_locations(const type_real x, const type_real z, const type_real xmin,
                     const type_real xmax, const type_real zmin,
                     const type_real zmax, const specfem::MPI::MPI *mpi);

int compute_nglob(const specfem::HostMirror3d<int> ibool);
} // namespace utilities
} // namespace specfem

#endif
