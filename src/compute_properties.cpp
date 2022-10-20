#include "../include/compute.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/shape_functions.h"
#include <Kokkos_Core.hpp>

specfem::compute::properties::properties(const int nspec, const int ngllz,
                                         const int ngllx)
    : rho(specfem::HostView3d<type_real>("specfem::mesh::rho", nspec, ngllz,
                                         ngllx)),
      mu(specfem::HostView3d<type_real>("specfem::mesh::mu", nspec, ngllz,
                                        ngllx)),
      kappa(specfem::HostView3d<type_real>("specfem::mesh::kappa", nspec, ngllz,
                                           ngllx)),
      qmu(specfem::HostView3d<type_real>("specfem::mesh::qmu", nspec, ngllz,
                                         ngllx)),
      qkappa(specfem::HostView3d<type_real>("specfem::mesh::qkappa", nspec,
                                            ngllz, ngllx)),
      rho_vp(specfem::HostView3d<type_real>("specfem::mesh::rho_vp", nspec,
                                            ngllz, ngllx)),
      rho_vs(specfem::HostView3d<type_real>("specfem::mesh::rho_vs", nspec,
                                            ngllz, ngllx)){};

specfem::compute::properties::properties(
    const specfem::HostView1d<int> kmato,
    const std::vector<specfem::material *> &materials, const int nspec,
    const int ngllx, const int ngllz) {

  // Setup mesh properties
  // UPDATEME::
  //           acoustic materials
  //           poroelastic materials
  //           axisymmetric materials
  //           anisotropic materials

  *this = specfem::compute::properties(nspec, ngllz, ngllx);

  Kokkos::parallel_for(
      "setup_mesh_properties",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix) {
        const int imat = kmato(ispec);
        utilities::return_holder holder = materials[imat]->get_properties();
        auto [rho, mu, kappa, qmu, qkappa] = std::make_tuple(
            holder.rho, holder.mu, holder.kappa, holder.qmu, holder.qkappa);
        this->rho(ispec, iz, ix) = rho;
        this->mu(ispec, iz, ix) = mu;
        this->kappa(ispec, iz, ix) = kappa;

        this->qmu(ispec, iz, ix) = qmu;
        this->qkappa(ispec, iz, ix) = qkappa;

        type_real vp = std::sqrt((kappa + mu) / rho);
        type_real vs = std::sqrt(mu / rho);

        this->rho_vp(ispec, iz, ix) = rho * vp;
        this->rho_vs(ispec, iz, ix) = rho * vs;
      });
}
