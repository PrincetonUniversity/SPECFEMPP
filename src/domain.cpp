#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/enums.h"
#include "../include/globals.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    specfem::compute::sources *sources, specfem::compute::receivers *receivers,
    specfem::quadrature::quadrature *quadx,
    specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field", nglob, ndim)),
      field_dot(specfem::kokkos::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(specfem::kokkos::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(specfem::kokkos::DeviceView2d<type_real>(
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
};

KOKKOS_IMPL_HOST_FUNCTION
void specfem::Domain::Elastic::assign_views() {

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  const int nglob = field.extent(0);
  const int ndim = field.extent(1);
  // Initialize views
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::initiaze_views",
      specfem::kokkos::DeviceMDrange<2>({ 0, 0 }, { nglob, ndim }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        this->field(iglob, idim) = 0;
        this->field_dot(iglob, idim) = 0;
        this->field_dot_dot(iglob, idim) = 0;
        this->rmass_inverse(iglob, idim) = 0;
      });

  // Compute the mass matrix
  specfem::kokkos::DeviceScatterView2d<type_real> results(rmass_inverse);
  auto wxgll = quadx->get_w();
  auto wzgll = quadz->get_w();
  auto rho = this->material_properties->rho;
  auto ispec_type = this->material_properties->ispec_type;
  auto jacobian = this->partial_derivatives->jacobian;
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::kokkos::DeviceMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
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

void specfem::Domain::Elastic::compute_stiffness_interaction() {

  const int ngllx = this->quadx->get_N();
  const int ngllz = this->quadz->get_N();
  const int ngllxz = ngllx * ngllz;
  const auto ibool = this->compute->ibool;
  const auto ispec_domain = this->ispec_domain;
  const auto xix = this->partial_derivatives->xix;
  const auto xiz = this->partial_derivatives->xiz;
  const auto gammax = this->partial_derivatives->gammax;
  const auto gammaz = this->partial_derivatives->gammaz;
  const auto jacobian = this->partial_derivatives->jacobian;
  const auto mu = this->material_properties->mu;
  const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();

  int scratch_size =
      specfem::kokkos::DeviceScratchView1d<type_real>::shmem_size(ngllx);
  scratch_size +=
      specfem::kokkos::DeviceScratchView1d<type_real>::shmem_size(ngllz);
  scratch_size +=
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllx);
  scratch_size +=
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllz);
  scratch_size +=
      2 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      3 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      4 *
      specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_forces",
      specfem::kokkos::DeviceTeam(this->nelem_domain, Kokkos::AUTO, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        // std::cout << team_member.league_rank() << std::endl;
        const int ispec = ispec_domain(team_member.league_rank());

        // Getting subviews for better readability
        // This has a small perfomance hit (It should be negligible)
        const auto sv_ibool =
            Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_xix =
            Kokkos::subview(xix, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_xiz =
            Kokkos::subview(xiz, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_gammax =
            Kokkos::subview(gammax, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_gammaz =
            Kokkos::subview(gammaz, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_jacobian =
            Kokkos::subview(jacobian, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_mu = Kokkos::subview(mu, ispec, Kokkos::ALL, Kokkos::ALL);
        const auto sv_lambdaplus2mu =
            Kokkos::subview(lambdaplus2mu, ispec, Kokkos::ALL, Kokkos::ALL);

        // Assign scratch views
        // Optional performance related scratch views
        specfem::kokkos::DeviceScratchView1d<type_real> s_wxgll(
            team_member.team_scratch(0), ngllx);
        specfem::kokkos::DeviceScratchView1d<type_real> s_wzgll(
            team_member.team_scratch(0), ngllz);
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempz(
            team_member.team_scratch(0), ngllz, ngllx);

        // Nacessary scratch views
        specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_xx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_xz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_zz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempx1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempz1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempx3(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::kokkos::DeviceScratchView2d<type_real> s_tempz3(
            team_member.team_scratch(0), ngllz, ngllx);

        // -------------Load into scratch memory----------------------------
        if (team_member.team_rank() == 0) {

          for (int ix = 0; ix < ngllx; ix++) {
            s_wxgll(ix) = wxgll(ix);
          }

          for (int iz = 0; iz < ngllz; iz++) {
            s_wzgll(iz) = wzgll(iz);
          }
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngllxz),
                             [=](const int xz) {
                               const int ix = xz % ngllz;
                               const int iz = xz / ngllz;
                               int iglob = sv_ibool(iz, ix);
                               s_tempx(iz, ix) = this->field(iglob, 0);
                               s_tempz(iz, ix) = this->field(iglob, 1);
                             });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
            [=](const int ij) {
              const int i = ij % ngllx;
              const int j = ij / ngllx;
              s_hprime_xx(j, i) = hprime_xx(j, i);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
            [=](const int ij) {
              const int i = ij % ngllz;
              const int j = ij / ngllz;
              s_hprime_zz(j, i) = hprime_zz(j, i);
            });
        //----------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;

              type_real sum_hprime_x1 = 0;
              type_real sum_hprime_x3 = 0;
              type_real sum_hprime_z1 = 0;
              type_real sum_hprime_z3 = 0;

              for (int l = 0; l < ngllx; l++) {
                sum_hprime_x1 += s_hprime_xx(ix, l) * s_tempx(iz, l);
                sum_hprime_x3 += s_hprime_xx(ix, l) * s_tempz(iz, l);
              }

              for (int l = 0; l < ngllz; l++) {
                sum_hprime_z1 += s_hprime_zz(iz, l) * s_tempx(l, ix);
                sum_hprime_z3 += s_hprime_zz(iz, l) * s_tempz(l, ix);
              }

              const type_real xixl = sv_xix(iz, ix);
              const type_real xizl = sv_xiz(iz, ix);
              const type_real gammaxl = sv_gammax(iz, ix);
              const type_real gammazl = sv_gammaz(iz, ix);
              const type_real jacobianl = sv_jacobian(iz, ix);

              const type_real duxdxl =
                  xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;
              const type_real duxdzl =
                  xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

              const type_real duzdxl =
                  xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;
              const type_real duzdzl =
                  xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

              const type_real duzdxl_plus_duxdzl = duzdxl + duxdzl;

              const type_real mul = sv_mu(iz, ix);
              const type_real lambdaplus2mul = sv_lambdaplus2mu(iz, ix);
              const type_real lambdal = lambdaplus2mul - 2.0 * mul;

              if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
                // P_SV case
                s_sigma_xx(iz, ix) = lambdaplus2mul * duxdxl + lambdal * duzdzl;
                s_sigma_zz(iz, ix) = lambdaplus2mul * duzdzl + lambdal * duxdxl;
                s_sigma_xz(iz, ix) = mul * duzdxl_plus_duxdzl;
              } else if (specfem::globals::simulation_wave ==
                         specfem::wave::sh) {
                // SH-case
                s_sigma_xx(iz, ix) =
                    mul * duxdxl; // would be sigma_xy in CPU-version
                s_sigma_xz(iz, ix) = mul * duxdzl; // sigma_zy
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              const type_real xixl = sv_xix(iz, ix);
              const type_real xizl = sv_xiz(iz, ix);
              const type_real jacobianl = sv_jacobian(iz, ix);
              s_tempx(iz, ix) = jacobianl * (s_sigma_xx(iz, ix) * xixl +
                                             s_sigma_xz(iz, ix) * xizl);
              s_tempz(iz, ix) = jacobianl * (s_sigma_xz(iz, ix) * xixl +
                                             s_sigma_zz(iz, ix) * xizl);
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;

              s_tempx1(iz, ix) = 0;
              s_tempz1(iz, ix) = 0;
              for (int l = 0; l < ngllx; l++) {
                s_tempx1(iz, ix) +=
                    s_wxgll(l) * s_hprime_xx(l, ix) * s_tempx(iz, l);
                s_tempz1(iz, ix) +=
                    s_wxgll(l) * s_hprime_xx(l, ix) * s_tempz(iz, l);
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              const type_real gammaxl = sv_gammax(iz, ix);
              const type_real gammazl = sv_gammaz(iz, ix);
              const type_real jacobianl = sv_jacobian(iz, ix);
              s_tempx(iz, ix) = jacobianl * (s_sigma_xx(iz, ix) * gammaxl +
                                             s_sigma_xz(iz, ix) * gammazl);
              s_tempz(iz, ix) = jacobianl * (s_sigma_xz(iz, ix) * gammaxl +
                                             s_sigma_zz(iz, ix) * gammazl);
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;

              s_tempx3(iz, ix) = 0;
              s_tempz3(iz, ix) = 0;
              for (int l = 0; l < ngllz; l++) {
                s_tempx3(iz, ix) +=
                    s_wzgll(l) * s_hprime_zz(l, iz) * s_tempx(l, ix);
                s_tempz3(iz, ix) +=
                    s_wzgll(l) * s_hprime_zz(l, iz) * s_tempz(l, ix);
              }
            });

        team_member.team_barrier();

        // assembles acceleration array
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              const int iglob = sv_ibool(iz, ix);
              const type_real sum_terms1 =
                  -1.0 * (s_wzgll(iz) * s_tempx1(iz, ix)) -
                  (s_wxgll(ix) * s_tempx3(iz, ix));
              const type_real sum_terms3 =
                  -1.0 * (s_wzgll(iz) * s_tempz1(iz, ix)) -
                  (s_wxgll(ix) * s_tempz3(iz, ix));
              Kokkos::single(Kokkos::PerThread(team_member), [=] {
                Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), sum_terms1);
                Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), sum_terms3);
              });
            });
      });

  Kokkos::fence();

  return;
}

KOKKOS_IMPL_HOST_FUNCTION
void specfem::Domain::Elastic::divide_mass_matrix() {

  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::kokkos::DeviceRange(0, nglob),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        this->field_dot_dot(iglob, 0) =
            this->field_dot_dot(iglob, 0) * this->rmass_inverse(iglob, 0);
        this->field_dot_dot(iglob, 1) =
            this->field_dot_dot(iglob, 1) * this->rmass_inverse(iglob, 1);
      });

  Kokkos::fence();

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

  Kokkos::fence();
  return;
}

// Copy the values of relevant field at the spectral element where receiver lies
// to a storage view (field)
KOKKOS_FUNCTION
void get_receiver_field_for_seismogram(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    specfem::kokkos::DeviceView3d<type_real> field,
    const specfem::seismogram::type type,
    const specfem::kokkos::DeviceView2d<int> sv_ibool,
    const specfem::Domain::Domain *domain) {

  const int ngllx = sv_ibool.extent(0);
  const int ngllz = sv_ibool.extent(1);
  const int ngllxz = ngllx * ngllz;
  specfem::kokkos::DeviceView2d<type_real> copy_field;

  switch (type) {
  // Get the displacement field
  case specfem::seismogram::displacement:
    copy_field = domain->get_field();
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
          const int ix = xz % ngllz;
          const int iz = xz / ngllz;
          const int iglob = sv_ibool(iz, ix);

          if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
            type_real field_v0 = copy_field(iglob, 0);
            field(0, iz, ix) = field_v0;
            type_real field_v1 = copy_field(iglob, 1);
            field(1, iz, ix) = field_v1;
          } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
            field(0, iz, ix) = copy_field(iglob, 0);
          }
        });
    break;
  // Get the velocity field
  case specfem::seismogram::velocity:
    copy_field = domain->get_field_dot();
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
          const int ix = xz % ngllz;
          const int iz = xz / ngllz;
          const int iglob = sv_ibool(iz, ix);

          if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
            type_real field_v0 = copy_field(iglob, 0);
            field(0, iz, ix) = field_v0;
            type_real field_v1 = copy_field(iglob, 1);
            field(1, iz, ix) = field_v1;
          } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
            field(0, iz, ix) = copy_field(iglob, 0);
          }
        });
    break;
  // Get the acceleration field
  case specfem::seismogram::acceleration:
    copy_field = domain->get_field_dot_dot();
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
          const int ix = xz % ngllz;
          const int iz = xz / ngllz;
          const int iglob = sv_ibool(iz, ix);

          if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
            field(0, iz, ix) = copy_field(iglob, 0);
            field(1, iz, ix) = copy_field(iglob, 1);
          } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
            field(0, iz, ix) = copy_field(iglob, 0);
          }
        });
    break;
  }

  return;
}

// Compute the seismogram using field view calculated using above function
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
  } // switch(type)

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
  const type_real test_value = this->field_dot(57619, 1);
  auto seismogram = Kokkos::subview(this->receivers->seismogram, isig_step,
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_seismogram",
      specfem::kokkos::DeviceTeam(nsigtype * nreceivers, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const int isigtype = team_member.league_rank() / nreceivers;
        const int irec = team_member.league_rank() % nreceivers;
        const int ispec = ispec_array(irec);
        if (ispec_type(ispec) == specfem::elements::elastic) {

          // Get seismogram field
          const specfem::seismogram::type type = seismogram_types(isigtype);
          const auto sv_ibool =
              Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);
          auto sv_field = Kokkos::subview(field, isigtype, irec, Kokkos::ALL,
                                          Kokkos::ALL, Kokkos::ALL);
          get_receiver_field_for_seismogram(team_member, sv_field, type,
                                            sv_ibool, this);

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
}
