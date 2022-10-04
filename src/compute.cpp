#include "../include/compute.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

specfem::compute::coordinates::coordinates(const int nspec, const int ngllz,
                                           const int ngllx) {
  this->xcor = specfem::HostView3d<type_real>("specfem::mesh::xcor", nspec,
                                              ngllz, ngllx);
  this->ycor = specfem::HostView3d<type_real>("specfem::mesh::ycor", nspec,
                                              ngllz, ngllx);
  this->xix =
      specfem::HostView3d<type_real>("specfem::mesh::xix", nspec, ngllz, ngllx);
  this->xiz =
      specfem::HostView3d<type_real>("specfem::mesh::xiz", nspec, ngllz, ngllx);
  this->gammax = specfem::HostView3d<type_real>("specfem::mesh::gammax", nspec,
                                                ngllz, ngllx);
  this->gammaz = specfem::HostView3d<type_real>("specfem::mesh::gammaz", nspec,
                                                ngllz, ngllx);
  this->jacobian = specfem::HostView3d<type_real>("specfem::mesh::jacobian",
                                                  nspec, ngllz, ngllx);

  return;
}

specfem::compute::properties::properties(const int nspec, const int ngllz,
                                         const int ngllx) {
  this->rho =
      specfem::HostView3d<type_real>("specfem::mesh::rho", nspec, ngllz, ngllx);
  this->mu =
      specfem::HostView3d<type_real>("specfem::mesh::mu", nspec, ngllz, ngllx);
  this->kappa = specfem::HostView3d<type_real>("specfem::mesh::kappa", nspec,
                                               ngllz, ngllx);
  this->qmu =
      specfem::HostView3d<type_real>("specfem::mesh::qmu", nspec, ngllz, ngllx);
  this->qkappa = specfem::HostView3d<type_real>("specfem::mesh::qkappa", nspec,
                                                ngllz, ngllx);
  this->rho_vp = specfem::HostView3d<type_real>("specfem::mesh::rho_vp", nspec,
                                                ngllz, ngllx);
  this->rho_vs = specfem::HostView3d<type_real>("specfem::mesh::rho_vs", nspec,
                                                ngllz, ngllx);

  return;
}
