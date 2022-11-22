#ifndef UTILS_H
#define UTILS_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace utilities {
struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

struct return_holder {
  type_real rho, mu, kappa, qmu, qkappa;
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
};

std::tuple<int, int, int, type_real, type_real>
locate(const specfem::HostView3d<int> ibool,
       const specfem::HostView2d<type_real> coord,
       const quadrature::quadrature &quadx, const quadrature::quadrature &quadz,
       const int nproc, const type_real x, const type_real z,
       const specfem::HostView3d<type_real> coorg,
       const specfem::HostView2d<int> knods, const int npgeo);
} // namespace utilities
} // namespace specfem

#endif
