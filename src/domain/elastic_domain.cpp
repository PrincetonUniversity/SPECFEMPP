#include "compute/interface.hpp"
#include "domain/interface.hpp"
#include "enums.h"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "mathematical_operators.h"
#include "quadrature.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(const int ndim, const int nglob)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field", nglob, ndim)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  return;
}

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    specfem::compute::sources *sources, specfem::compute::receivers *receivers,
    specfem::quadrature::quadrature *quadx,
    specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field", nglob, ndim)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)),
      compute(compute), material_properties(material_properties),
      partial_derivatives(partial_derivatives), sources(sources),
      receivers(receivers), quadx(quadx), quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  this->assign_views();

  return;
};

KOKKOS_IMPL_HOST_FUNCTION
void specfem::Domain::Elastic::assign_views() {

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  const int nglob = field.extent(0);

  // Inverse of mass matrix
  //----------------------------------------------------------------------------
  // Initialize views
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>({ 0, 0 },
                                                               { nglob, ndim }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        this->field(iglob, idim) = 0;
        this->field_dot(iglob, idim) = 0;
        this->field_dot_dot(iglob, idim) = 0;
        this->rmass_inverse(iglob, idim) = 0;
      });

  // Compute the mass matrix
  specfem::kokkos::DeviceScatterView2d<type_real, Kokkos::LayoutLeft> results(
      rmass_inverse);
  auto wxgll = quadx->get_w();
  auto wzgll = quadz->get_w();
  auto rho = this->material_properties->rho;
  auto ispec_type = this->material_properties->ispec_type;
  auto jacobian = this->partial_derivatives->jacobian;
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::kokkos::DeviceMDrange<3, Kokkos::Iterate::Left>(
          { 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, iz, ix);
        auto access = results.access();
        if (ispec_type(ispec) == specfem::elements::elastic) {
          access(iglob, 0) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
          access(iglob, 1) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);

  // invert the mass matrix
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::Invert_mass_matrix",
      specfem::kokkos::DeviceRange(0, nglob),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        if (rmass_inverse(iglob, 0) > 0.0) {
          rmass_inverse(iglob, 0) = 1.0 / rmass_inverse(iglob, 0);
          rmass_inverse(iglob, 1) = 1.0 / rmass_inverse(iglob, 1);
        } else {
          rmass_inverse(iglob, 0) = 1.0;
          rmass_inverse(iglob, 1) = 1.0;
        }
      });
  // ----------------------------------------------------------------------

  // Domain type
  // ----------------------------------------------------------------------
  this->nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->h_ispec_type(ispec) ==
        specfem::elements::elastic) {
      this->nelem_domain++;
    }
  }

  this->ispec_domain = specfem::kokkos::DeviceView1d<int>(
      "specfem::Domain::Elastic::ispec_domain", this->nelem_domain);
  this->h_ispec_domain = Kokkos::create_mirror_view(ispec_domain);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->h_ispec_type(ispec) ==
        specfem::elements::elastic) {
      this->h_ispec_domain(index) = ispec;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  return;
}

void specfem::Domain::Elastic::sync_field(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_field_dot(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_field_dot_dot(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_rmass_inverse(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

// Specialized kernel when NGLLX == NGLLZ
// This kernel is templated for compiler optimizations.
// Specific instances of this kernel should be instantiated inside the kernel
// calling routine
template <int NGLL>
void specfem::Domain::Elastic::compute_stiffness_interaction() {

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto xix = this->partial_derivatives->xix;
  const auto xiz = this->partial_derivatives->xiz;
  const auto gammax = this->partial_derivatives->gammax;
  const auto gammaz = this->partial_derivatives->gammaz;
  const auto ibool = this->compute->ibool;
  const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
  const auto mu = this->material_properties->mu;
  const auto jacobian = this->partial_derivatives->jacobian;
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto ispec_domain = this->ispec_domain;
  const auto field = this->field;
  auto field_dot_dot = this->field_dot_dot;

  constexpr int NGLL2 = NGLL * NGLL;
  constexpr type_real NGLL_INV = 1.0 / NGLL;

  static_assert(NGLL2 == NGLL * NGLL);

  assert(hprime_zz.extent(0) == NGLL);
  assert(hprime_xx.extent(0) == NGLL);
  assert(hprime_zz.extent(1) == NGLL);
  assert(hprime_xx.extent(1) == NGLL);
  assert(xix.extent(1) == NGLL);
  assert(xix.extent(2) == NGLL);
  assert(xiz.extent(1) == NGLL);
  assert(xiz.extent(2) == NGLL);
  assert(gammax.extent(1) == NGLL);
  assert(gammax.extent(2) == NGLL);
  assert(gammaz.extent(1) == NGLL);
  assert(gammaz.extent(2) == NGLL);
  assert(ibool.extent(1) == NGLL);
  assert(ibool.extent(2) == NGLL);
  assert(lambdaplus2mu.extent(1) == NGLL);
  assert(lambdaplus2mu.extent(2) == NGLL);
  assert(mu.extent(1) == NGLL);
  assert(mu.extent(1) == NGLL);

  int scratch_size =
      10 * specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL,
                                                      NGLL>::shmem_size();

  scratch_size +=
      specfem::kokkos::StaticDeviceScratchView2d<int, NGLL, NGLL>::shmem_size();

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const int ispec = ispec_domain(team_member.league_rank());

        // Assign scratch views
        // Assign scratch views for views that are required by every thread
        // during summations
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_hprime_xx(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_hprime_zz(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_hprimewgll_xx(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_hprimewgll_zz(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<int, NGLL, NGLL> s_iglob(
            team_member.team_scratch(0));

        // Temporary scratch arrays used in calculation of integrals
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_fieldx(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_fieldz(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_temp1(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_temp2(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_temp3(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
            s_temp4(team_member.team_scratch(0));

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;
              const int iglob = ibool(ispec, iz, ix);
              s_fieldx(iz, ix) = field(iglob, 0);
              s_fieldz(iz, ix) = field(iglob, 1);
              s_temp1(iz, ix) = 0.0;
              s_temp2(iz, ix) = 0.0;
              s_temp3(iz, ix) = 0.0;
              s_temp4(iz, ix) = 0.0;
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
              s_hprimewgll_xx(ix, iz) = wxgll(iz) * hprime_xx(iz, ix);
              s_hprimewgll_zz(ix, iz) = wzgll(iz) * hprime_zz(iz, ix);
              s_iglob(iz, ix) = iglob;
            });

        team_member.team_barrier();

        specfem::mathematical_operators::compute_gradients_2D(
            team_member, ispec, xix, xiz, gammax, gammaz, s_hprime_xx,
            s_hprime_zz, s_fieldx, s_fieldz, s_temp1, s_temp2, s_temp3,
            s_temp4);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;

              const type_real lambdaplus2mul = lambdaplus2mu(ispec, iz, ix);
              const type_real mul = mu(ispec, iz, ix);
              const type_real lambdal = lambdaplus2mul - 2.0 * mul;

              const type_real xixl = xix(ispec, iz, ix);
              const type_real xizl = xiz(ispec, iz, ix);
              const type_real gammaxl = gammax(ispec, iz, ix);
              const type_real gammazl = gammaz(ispec, iz, ix);
              const type_real jacobianl = jacobian(ispec, iz, ix);

              type_real sigma_xx, sigma_zz, sigma_xz;

              if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
                // P_SV case
                // sigma_xx
                sigma_xx = lambdaplus2mul * s_temp1(iz, ix) +
                           lambdal * s_temp4(iz, ix);

                // sigma_zz
                sigma_zz = lambdaplus2mul * s_temp4(iz, ix) +
                           lambdal * s_temp1(iz, ix);

                // sigma_xz
                sigma_xz = mul * (s_temp3(iz, ix) + s_temp2(iz, ix));
              } else if (specfem::globals::simulation_wave ==
                         specfem::wave::sh) {
                // SH-case
                // sigma_xx
                sigma_xx = mul * s_temp1(iz, ix); // would be sigma_xy in
                                                  // CPU-version

                // sigma_xz
                sigma_xz = mul * s_temp2(iz, ix); // sigma_zy
              }

              s_temp1(iz, ix) = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
              s_temp2(iz, ix) = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
              s_temp3(iz, ix) =
                  jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
              s_temp4(iz, ix) =
                  jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);
            });

        team_member.team_barrier();

        specfem::mathematical_operators::add_contributions(
            team_member, wxgll, wzgll, s_hprimewgll_xx, s_hprimewgll_zz,
            s_iglob, s_temp1, s_temp2, s_temp3, s_temp4, field_dot_dot);
      });

  Kokkos::fence();

  return;
}

void specfem::Domain::Elastic::compute_stiffness_interaction() {

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto xix = this->partial_derivatives->xix;
  const auto xiz = this->partial_derivatives->xiz;
  const auto gammax = this->partial_derivatives->gammax;
  const auto gammaz = this->partial_derivatives->gammaz;
  const auto ibool = this->compute->ibool;
  const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
  const auto mu = this->material_properties->mu;
  const auto jacobian = this->partial_derivatives->jacobian;
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto ispec_domain = this->ispec_domain;
  const auto field = this->field;
  auto field_dot_dot = this->field_dot_dot;

  const int ngllz = xix.extent(1);
  const int ngllx = xiz.extent(2);

  const int ngll2 = ngllz * ngllx;

  int scratch_size =
      2 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllx);

  scratch_size +=
      2 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllz);

  scratch_size +=
      7 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllx);

  scratch_size +=
      specfem::kokkos::DeviceScratchView2d<int>::shmem_size(ngllz, ngllx);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const int ispec = ispec_domain(team_member.league_rank());

        // Assign scratch views
        // Assign scratch views for views that are required by every thread
        // during summations
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::kokkos::DeviceScratchView2d<int> s_iglob(
            team_member.team_scratch(0), ngllz, ngllx);

        // Temporary scratch arrays used in calculation of integrals
        specfem::kokkos::DeviceScratchView2d<type_real> s_fieldx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_fieldz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_temp1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_temp2(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_temp3(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_temp4(
            team_member.team_scratch(0), ngllz, ngllx);

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
              s_hprimewgll_xx(ix, iz) = wxgll(iz) * hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
              s_hprimewgll_zz(ix, iz) = wzgll(iz) * hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngll2),
                             [&](const int xz) {
                               int iz, ix;
                               sub2ind(xz, ngllx, iz, ix);
                               const int iglob = ibool(ispec, iz, ix);
                               s_fieldx(iz, ix) = field(iglob, 0);
                               s_fieldz(iz, ix) = field(iglob, 1);
                               s_temp1(iz, ix) = 0.0;
                               s_temp2(iz, ix) = 0.0;
                               s_temp3(iz, ix) = 0.0;
                               s_temp4(iz, ix) = 0.0;
                               s_iglob(iz, ix) = iglob;
                             });

        // ------------------------------------------------------------------

        team_member.team_barrier();

        specfem::mathematical_operators::compute_gradients_2D(
            team_member, ispec, xix, xiz, gammax, gammaz, s_hprime_xx,
            s_hprime_zz, s_fieldx, s_fieldz, s_temp1, s_temp2, s_temp3,
            s_temp4);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngll2), [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);

              const type_real lambdaplus2mul = lambdaplus2mu(ispec, iz, ix);
              const type_real mul = mu(ispec, iz, ix);
              const type_real lambdal = lambdaplus2mul - 2.0 * mul;

              const type_real xixl = xix(ispec, iz, ix);
              const type_real xizl = xiz(ispec, iz, ix);
              const type_real gammaxl = gammax(ispec, iz, ix);
              const type_real gammazl = gammaz(ispec, iz, ix);
              const type_real jacobianl = jacobian(ispec, iz, ix);

              type_real sigma_xx, sigma_zz, sigma_xz;

              if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
                // P_SV case
                // sigma_xx
                sigma_xx = lambdaplus2mul * s_temp1(iz, ix) +
                           lambdal * s_temp4(iz, ix);

                // sigma_zz
                sigma_zz = lambdaplus2mul * s_temp4(iz, ix) +
                           lambdal * s_temp1(iz, ix);

                // sigma_xz
                sigma_xz = mul * (s_temp3(iz, ix) + s_temp2(iz, ix));
              } else if (specfem::globals::simulation_wave ==
                         specfem::wave::sh) {
                // SH-case
                // sigma_xx
                sigma_xx = mul * s_temp1(iz, ix); // would be sigma_xy in
                                                  // CPU-version

                // sigma_xz
                sigma_xz = mul * s_temp2(iz, ix); // sigma_zy
              }

              s_temp1(iz, ix) = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
              s_temp2(iz, ix) = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
              s_temp3(iz, ix) =
                  jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
              s_temp4(iz, ix) =
                  jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);
            });

        team_member.team_barrier();

        specfem::mathematical_operators::add_contributions(
            team_member, wxgll, wzgll, s_hprimewgll_xx, s_hprimewgll_zz,
            s_iglob, s_temp1, s_temp2, s_temp3, s_temp4, field_dot_dot);
      });

  Kokkos::fence();

  return;
}

void specfem::Domain::Elastic::compute_stiffness_interaction_calling_routine() {

  const int ngllx = this->partial_derivatives->xix.extent(1);
  const int ngllz = this->partial_derivatives->xix.extent(2);

  if (ngllx == 5 && (ngllx == ngllz)) {
    this->compute_stiffness_interaction<5>();
  } else if (ngllx == 8 && (ngllx == ngllz)) {
    this->compute_stiffness_interaction<8>();
  } else {
    this->compute_stiffness_interaction();
  }

  return;
}

KOKKOS_IMPL_HOST_FUNCTION
void specfem::Domain::Elastic::divide_mass_matrix() {

  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::kokkos::DeviceRange(0, ndim * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        this->field_dot_dot(iglob, idim) =
            this->field_dot_dot(iglob, idim) * this->rmass_inverse(iglob, idim);
      });

  // Kokkos::fence();

  return;
}

void specfem::Domain::Elastic::compute_source_interaction(
    const type_real timeval) {

  const int nsources = this->sources->source_array.extent(0);
  const int ngllz = this->sources->source_array.extent(1);
  const int ngllx = this->sources->source_array.extent(2);
  const int ngllxz = ngllx * ngllz;
  const auto ispec_array = this->sources->ispec_array;
  const auto ispec_type = this->material_properties->ispec_type;
  const auto stf_array = this->sources->stf_array;
  const auto source_array = this->sources->source_array;
  const auto ibool = this->compute->ibool;

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_source_interaction",
      specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int isource = team_member.league_rank();
        int ispec = ispec_array(isource);
        auto sv_ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

        if (ispec_type(ispec) == specfem::elements::elastic) {

          type_real stf;

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team_member, 1),
              [=](const int &, type_real &lsum) {
                lsum = stf_array(isource).T->compute(timeval);
              },
              stf);

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;
                int iglob = sv_ibool(iz, ix);

                if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
                  const type_real accelx =
                      source_array(isource, iz, ix, 0) * stf;
                  const type_real accelz =
                      source_array(isource, iz, ix, 1) * stf;
                  Kokkos::single(Kokkos::PerThread(team_member), [=] {
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), accelz);
                  });
                } else if (specfem::globals::simulation_wave ==
                           specfem::wave::sh) {
                  const type_real accelx =
                      source_array(isource, iz, ix, 0) * stf;
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                }
              });
        }
      });

  // Kokkos::fence();
  return;
}

// Compute the seismogram using field view
KOKKOS_FUNCTION
void compute_receiver_seismogram(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    specfem::kokkos::DeviceView1d<type_real> sv_seismogram,
    const specfem::kokkos::DeviceView3d<type_real> field,
    const specfem::seismogram::type type,
    const specfem::kokkos::DeviceView3d<type_real> sv_receiver_array,
    const type_real cos_irec, const type_real sin_irec) {

  const int ngllx = sv_receiver_array.extent(0);
  const int ngllz = sv_receiver_array.extent(1);
  const int ngllxz = ngllx * ngllz;
  switch (type) {
  case specfem::seismogram::displacement:
  case specfem::seismogram::velocity:
  case specfem::seismogram::acceleration:

    type_real vx = 0.0;
    type_real vz = 0.0;

    if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, ngllxz),
          [=](const int xz, type_real &l_vx) {
            const int ix = xz % ngllz;
            const int iz = xz / ngllz;
            const type_real hlagrange = sv_receiver_array(iz, ix, 0);
            const type_real field_v = field(0, iz, ix);

            l_vx += field_v * hlagrange;
          },
          vx);
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, ngllxz),
          [=](const int xz, type_real &l_vz) {
            const int ix = xz % ngllz;
            const int iz = xz / ngllz;
            const type_real hlagrange = sv_receiver_array(iz, ix, 0);
            const type_real field_v = field(1, iz, ix);

            l_vz += field_v * hlagrange;
          },
          vz);
    } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, ngllxz),
          [=](const int xz, type_real &l_vx) {
            const int ix = xz % ngllz;
            const int iz = xz / ngllz;
            const type_real hlagrange = sv_receiver_array(iz, ix, 0);
            const type_real field_v = field(0, iz, ix);

            l_vx += field_v * hlagrange;
          },
          vx);
    }

    Kokkos::single(Kokkos::PerTeam(team_member), [=] {
      if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
        sv_seismogram(0) = cos_irec * vx + sin_irec * vz;
        sv_seismogram(1) = sin_irec * vx + cos_irec * vz;
      } else if ((specfem::globals::simulation_wave == specfem::wave::sh)) {
        sv_seismogram(0) = cos_irec * vx + sin_irec * vz;
        sv_seismogram(1) = 0;
      }
    });

    break;
  }

  return;
}

void specfem::Domain::Elastic::compute_seismogram(const int isig_step) {

  const auto seismogram_types = this->receivers->seismogram_types;
  const int nsigtype = seismogram_types.extent(0);
  const int nreceivers = this->receivers->receiver_array.extent(0);
  const auto ispec_array = this->receivers->ispec_array;
  const auto ispec_type = this->material_properties->ispec_type;
  const auto receiver_array = this->receivers->receiver_array;
  const auto ibool = this->compute->ibool;
  const auto cos_recs = this->receivers->cos_recs;
  const auto sin_recs = this->receivers->sin_recs;
  auto field = this->receivers->field;
  const int ngllx = ibool.extent(1);
  const int ngllz = ibool.extent(2);
  const int ngllxz = ngllx * ngllz;
  auto seismogram = Kokkos::subview(this->receivers->seismogram, isig_step,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  specfem::kokkos::DeviceView2d<type_real> copy_field;

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_seismogram",
      specfem::kokkos::DeviceTeam(nsigtype * nreceivers, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const int isigtype = team_member.league_rank() / nreceivers;
        const int irec = team_member.league_rank() % nreceivers;
        const int ispec = ispec_array(irec);
        if (ispec_type(ispec) == specfem::elements::elastic) {

          const specfem::seismogram::type type = seismogram_types(isigtype);
          const auto sv_ibool =
              Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);
          auto sv_field = Kokkos::subview(field, isigtype, irec, Kokkos::ALL,
                                          Kokkos::ALL, Kokkos::ALL);
          // Get seismogram field
          // ----------------------------------------------------------------
          switch (type) {
          // Get the displacement field
          case specfem::seismogram::displacement:
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, ngllxz),
                [=](const int xz) {
                  const int ix = xz % ngllz;
                  const int iz = xz / ngllz;
                  const int iglob = sv_ibool(iz, ix);

                  if (specfem::globals::simulation_wave ==
                      specfem::wave::p_sv) {
                    sv_field(0, iz, ix) = this->field(iglob, 0);
                    sv_field(1, iz, ix) = this->field(iglob, 1);
                  } else if (specfem::globals::simulation_wave ==
                             specfem::wave::sh) {
                    sv_field(0, iz, ix) = this->field(iglob, 0);
                  }
                });
            break;
          // Get the velocity field
          case specfem::seismogram::velocity:
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, ngllxz),
                [=](const int xz) {
                  const int ix = xz % ngllz;
                  const int iz = xz / ngllz;
                  const int iglob = sv_ibool(iz, ix);

                  if (specfem::globals::simulation_wave ==
                      specfem::wave::p_sv) {
                    sv_field(0, iz, ix) = this->field_dot(iglob, 0);
                    sv_field(1, iz, ix) = this->field_dot(iglob, 1);
                  } else if (specfem::globals::simulation_wave ==
                             specfem::wave::sh) {
                    sv_field(0, iz, ix) = this->field_dot(iglob, 0);
                  }
                });
            break;
          // Get the acceleration field
          case specfem::seismogram::acceleration:
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, ngllxz),
                [=](const int xz) {
                  const int ix = xz % ngllz;
                  const int iz = xz / ngllz;
                  const int iglob = sv_ibool(iz, ix);

                  if (specfem::globals::simulation_wave ==
                      specfem::wave::p_sv) {
                    sv_field(0, iz, ix) = this->field_dot_dot(iglob, 0);
                    sv_field(1, iz, ix) = this->field_dot_dot(iglob, 1);
                  } else if (specfem::globals::simulation_wave ==
                             specfem::wave::sh) {
                    sv_field(0, iz, ix) = this->field_dot_dot(iglob, 0);
                  }
                });
            break;
          }
          //-------------------------------------------------------------------

          // compute seismograms
          const auto sv_receiver_array = Kokkos::subview(
              receiver_array, irec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
          const type_real cos_irec = cos_recs(irec);
          const type_real sin_irec = sin_recs(irec);
          auto sv_seismogram =
              Kokkos::subview(seismogram, isigtype, irec, Kokkos::ALL);
          compute_receiver_seismogram(team_member, sv_seismogram, sv_field,
                                      type, sv_receiver_array, cos_irec,
                                      sin_irec);
        }
      });

  // Kokkos::fence();
}
