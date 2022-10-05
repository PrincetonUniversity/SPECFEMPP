#ifndef COMPUTE_H
#define COMPUTE_H

#include "../include/kokkos_abstractions.h"

namespace specfem {
namespace compute {

struct coordinates {
  specfem::HostView3d<type_real> xcor, ycor, xix, xiz, gammax, gammaz, jacobian;
  coordinates(){};
  coordinates(const int nspec, const int ngllz, const int ngllx);
};

struct properties {
  specfem::HostView3d<type_real> rho, mu, kappa, qmu, qkappa, rho_vp, rho_vs;
  properties(){};
  properties(const int nspec, const int ngllz, const int ngllx);
};

struct compute {
  specfem::HostView3d<int> ibool;
  specfem::compute::coordinates coordinates;
  specfem::compute::properties properties;
  compute(){};
  compute(const int nspec, const int ngllx, const int ngllz);
};

} // namespace compute
} // namespace specfem

#endif
