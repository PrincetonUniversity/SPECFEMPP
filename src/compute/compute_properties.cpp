#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

specfem::compute::properties::properties(const int nspec, const int ngllz,
                                         const int ngllx)
    : rho(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::rho", nspec, ngllz, ngllx)),
      mu(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::mu", nspec, ngllz, ngllx)),
      kappa(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::kappa", nspec, ngllz, ngllx)),
      qmu(specfem::kokkos::HostView3d<type_real>(
          "specfem::compute::properties::qmu", nspec, ngllz, ngllx)),
      qkappa(specfem::kokkos::HostView3d<type_real>(
          "specfem::compute::properties::qkappa", nspec, ngllz, ngllx)),
      rho_vp(specfem::kokkos::HostView3d<type_real>(
          "specfem::compute::properties::rho_vp", nspec, ngllz, ngllx)),
      rho_vs(specfem::kokkos::HostView3d<type_real>(
          "specfem::compute::properties::rho_vs", nspec, ngllz, ngllx)),
      lambdaplus2mu(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::lambdaplus2mu", nspec, ngllz, ngllx)),
      ispec_type(specfem::kokkos::DeviceView1d<specfem::enums::element::type>(
          "specfem::compute::properties::ispec_type", nspec)),
      rho_inverse(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::rho_inverse", nspec, ngllz, ngllx)),
      lambdaplus2mu_inverse(specfem::kokkos::DeviceView3d<type_real>(
          "specfem::compute::properties::lambdaplus2mu_inverse", nspec, ngllz,
          ngllx)) {

  h_rho = Kokkos::create_mirror_view(rho);
  h_mu = Kokkos::create_mirror_view(mu);
  h_kappa = Kokkos::create_mirror_view(kappa);
  h_lambdaplus2mu = Kokkos::create_mirror_view(lambdaplus2mu);
  h_ispec_type = Kokkos::create_mirror_view(ispec_type);
  h_rho_inverse = Kokkos::create_mirror_view(rho_inverse);
  h_lambdaplus2mu_inverse = Kokkos::create_mirror_view(lambdaplus2mu_inverse);
};

specfem::compute::properties::properties(
    const specfem::kokkos::HostView1d<int> kmato,
    const std::vector<std::shared_ptr<specfem::material::material> > &materials,
    const int nspec, const int ngllx, const int ngllz) {

  // Setup compute::properties properties
  // UPDATEME::
  //           acoustic materials
  //           poroelastic materials
  //           axisymmetric materials
  //           anisotropic materials

  *this = specfem::compute::properties(nspec, ngllz, ngllx);

  Kokkos::parallel_for(
      "specfem::compute::properties::properties",
      specfem::kokkos::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      [=](const int ispec, const int iz, const int ix) {
        const int imat = kmato(ispec);
        utilities::return_holder holder = materials[imat]->get_properties();
        auto [rho, mu, kappa, qmu, qkappa, lambdaplus2mu] =
            std::make_tuple(holder.rho, holder.mu, holder.kappa, holder.qmu,
                            holder.qkappa, holder.lambdaplus2mu);
        this->h_rho(ispec, iz, ix) = rho;
        this->h_mu(ispec, iz, ix) = mu;
        this->h_kappa(ispec, iz, ix) = kappa;

        this->qmu(ispec, iz, ix) = qmu;
        this->qkappa(ispec, iz, ix) = qkappa;

        type_real vp = std::sqrt((kappa + mu) / rho);
        type_real vs = std::sqrt(mu / rho);

        this->rho_vp(ispec, iz, ix) = rho * vp;
        this->rho_vs(ispec, iz, ix) = rho * vs;
        this->h_rho_inverse(ispec, iz, ix) = 1.0 / rho;
        this->h_lambdaplus2mu_inverse(ispec, iz, ix) = 1.0 / lambdaplus2mu;
        this->h_lambdaplus2mu(ispec, iz, ix) = lambdaplus2mu;
      });

  Kokkos::parallel_for(
      "setup_compute::properties_ispec", specfem::kokkos::HostRange(0, nspec),
      [=](const int ispec) {
        const int imat = kmato(ispec);
        this->h_ispec_type(ispec) = materials[imat]->get_ispec_type();
      });

  this->sync_views();
}

void specfem::compute::properties::sync_views() {
  Kokkos::deep_copy(rho, h_rho);
  Kokkos::deep_copy(mu, h_mu);
  Kokkos::deep_copy(kappa, h_kappa);
  Kokkos::deep_copy(lambdaplus2mu, h_lambdaplus2mu);
  Kokkos::deep_copy(ispec_type, h_ispec_type);
  Kokkos::deep_copy(rho_inverse, h_rho_inverse);
  Kokkos::deep_copy(lambdaplus2mu_inverse, h_lambdaplus2mu_inverse);

  return;
}
