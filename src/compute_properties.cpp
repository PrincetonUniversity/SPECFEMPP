#include "../include/compute.h"
#include "../include/config.h"
#include "../include/jacobian.h"
#include "../include/kokkos_abstractions.h"
#include "../include/shape_functions.h"
#include <Kokkos_Core.hpp>

specfem::compute::properties::properties(const int nspec, const int ngllz,
                                         const int ngllx)
    : rho(specfem::HostView3d<type_real>("specfem::compute::properties::rho",
                                         nspec, ngllz, ngllx)),
      mu(specfem::HostView3d<type_real>("specfem::compute::properties::mu",
                                        nspec, ngllz, ngllx)),
      kappa(specfem::HostView3d<type_real>(
          "specfem::compute::properties::kappa", nspec, ngllz, ngllx)),
      qmu(specfem::HostView3d<type_real>("specfem::compute::properties::qmu",
                                         nspec, ngllz, ngllx)),
      qkappa(specfem::HostView3d<type_real>(
          "specfem::compute::properties::qkappa", nspec, ngllz, ngllx)),
      rho_vp(specfem::HostView3d<type_real>(
          "specfem::compute::properties::rho_vp", nspec, ngllz, ngllx)),
      rho_vs(specfem::HostView3d<type_real>(
          "specfem::compute::properties::rho_vs", nspec, ngllz, ngllx)),
      lambdaplus2mu(specfem::HostView3d<type_real>(
          "specfem::compute::properties::lambdaplus2mu", nspec, ngllz, ngllx)),
      ispec_type(specfem::HostView1d<element_type>(
          "specfem::compute::properties::ispec_type", nspec)){};

specfem::compute::properties::properties(
    const specfem::HostView1d<int> kmato,
    const std::vector<specfem::material *> &materials, const int nspec,
    const int ngllx, const int ngllz) {

  // Setup compute::properties properties
  // UPDATEME::
  //           acoustic materials
  //           poroelastic materials
  //           axisymmetric materials
  //           anisotropic materials

  *this = specfem::compute::properties(nspec, ngllz, ngllx);

  Kokkos::parallel_for(
      "setup_compute::properties_properties",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      [=](const int ispec, const int iz, const int ix) {
        const int imat = kmato(ispec);
        utilities::return_holder holder = materials[imat]->get_properties();
        auto [rho, mu, kappa, qmu, qkappa, lambdaplus2mu] =
            std::make_tuple(holder.rho, holder.mu, holder.kappa, holder.qmu,
                            holder.qkappa, holder.lambdaplus2mu);
        this->rho(ispec, iz, ix) = rho;
        this->mu(ispec, iz, ix) = mu;
        this->kappa(ispec, iz, ix) = kappa;

        this->qmu(ispec, iz, ix) = qmu;
        this->qkappa(ispec, iz, ix) = qkappa;

        type_real vp = std::sqrt((kappa + mu) / rho);
        type_real vs = std::sqrt(mu / rho);

        this->rho_vp(ispec, iz, ix) = rho * vp;
        this->rho_vs(ispec, iz, ix) = rho * vs;
        this->lambdaplus2mu(ispec, iz, ix) = lambdaplus2mu;
      });

  Kokkos::parallel_for("setup_compute::properties_ispec",
                       specfem::HostRange(0, nspec), [=](const int ispec) {
                         const int imat = kmato(ispec);
                         this->ispec_type(ispec) =
                             materials[imat]->get_ispec_type();
                       });
}
