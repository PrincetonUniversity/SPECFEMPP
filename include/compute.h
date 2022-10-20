#ifndef COMPUTE_H
#define COMPUTE_H

#include "../include/kokkos_abstractions.h"
#include "../include/material.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

struct coordinates {
  specfem::HostView3d<type_real> xcor, ycor, xix, xiz, gammax, gammaz, jacobian;
  coordinates(){};
  coordinates(const int nspec, const int ngllz, const int ngllx);
  coordinates(const specfem::HostView2d<type_real> coorg,
              const specfem::HostView2d<int> knods,
              const quadrature::quadrature &quadx,
              const quadrature::quadrature &quadz);
};

struct properties {
  specfem::HostView3d<type_real> rho, mu, kappa, qmu, qkappa, rho_vp, rho_vs;
  properties(){};
  properties(const int nspec, const int ngllz, const int ngllx);
  properties(const specfem::HostView1d<int> kmato,
             const std::vector<specfem::material *> &materials, const int nspec,
             const int ngllx, const int ngllz);
};

struct compute {
  specfem::HostView3d<int> ibool;
  specfem::compute::coordinates coordinates;
  specfem::compute::properties properties;
  compute(){};
  compute(const int nspec, const int ngllx, const int ngllz);
  compute(const specfem::HostView2d<type_real> coorg,
          const specfem::HostView2d<int> knods,
          const specfem::HostView1d<int> kmato,
          const quadrature::quadrature &quadx,
          const quadrature::quadrature &quadz,
          const std::vector<specfem::material *> &materials);
};

} // namespace compute
} // namespace specfem

#endif
