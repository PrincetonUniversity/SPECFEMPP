#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    specfem::compute::sources *sources, quadrature::quadrature *quadx,
    quadrature::quadrature *quadz)
    : field(specfem::DeviceView2d<type_real>("specfem::Domain::Elastic::field",
                                             nglob, ndim)),
      field_dot(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)),
      compute(compute), material_properties(material_properties),
      partial_derivatives(partial_derivatives), sources(sources), quadx(quadx),
      quadz(quadz) {

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
    if (material_properties->h_ispec_type(ispec) == elastic) {
      this->nelem_domain++;
    }
  }

  this->ispec_domain = specfem::DeviceView1d<int>(
      "specfem::Domain::Elastic::ispec_domain", this->nelem_domain);
  this->h_ispec_domain = Kokkos::create_mirror_view(ispec_domain);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->h_ispec_type(ispec) == elastic) {
      this->h_ispec_domain(index) = ispec;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  return;
};

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
      specfem::DeviceMDrange<2>({ 0, 0 }, { nglob, ndim }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        this->field(iglob, idim) = 0;
        this->field_dot(iglob, idim) = 0;
        this->field_dot_dot(iglob, idim) = 0;
        this->rmass_inverse(iglob, idim) = 0;
      });

  // Compute the mass matrix
  specfem::DeviceScatterView2d<type_real> results(rmass_inverse);
  auto wxgll = quadx->get_w();
  auto wzgll = quadz->get_w();
  auto rho = this->material_properties->rho;
  auto ispec_type = this->material_properties->ispec_type;
  auto jacobian = this->partial_derivatives->jacobian;
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::DeviceMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, iz, ix);
        auto access = results.access();
        if (ispec_type(ispec) == elastic) {
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
      specfem::DeviceRange(0, nglob), KOKKOS_CLASS_LAMBDA(const int iglob) {
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

  int scratch_size = specfem::DeviceScratchView1d<type_real>::shmem_size(ngllx);
  scratch_size += specfem::DeviceScratchView1d<type_real>::shmem_size(ngllz);
  scratch_size +=
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllx);
  scratch_size +=
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllz);
  scratch_size +=
      2 * specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      3 * specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      4 * specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllz);

  // int catch_scratch_size = specfem::HostTeam(this->nelem_domain,
  // Kokkos::AUTO, ngllx).scratch_size_max(0); int hbmem_scratch_size =
  // specfem::HostTeam(this->nelem_domain, Kokkos::AUTO,
  // ngllx).scratch_size_max(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_forces",
      specfem::DeviceTeam(this->nelem_domain, 32, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(const specfem::DeviceTeam::member_type &team_member) {
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
        specfem::DeviceScratchView1d<type_real> s_wxgll(
            team_member.team_scratch(0), ngllx);
        specfem::DeviceScratchView1d<type_real> s_wzgll(
            team_member.team_scratch(0), ngllz);
        specfem::DeviceScratchView2d<type_real> s_hprime_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::DeviceScratchView2d<type_real> s_hprime_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::DeviceScratchView2d<type_real> s_tempx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_tempz(
            team_member.team_scratch(0), ngllz, ngllx);

        // Nacessary scratch views
        specfem::DeviceScratchView2d<type_real> s_sigma_xx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_sigma_xz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_sigma_zz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_tempx1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_tempz1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_tempx3(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::DeviceScratchView2d<type_real> s_tempz3(
            team_member.team_scratch(0), ngllz, ngllx);

        // if (team_member.team_rank() == 0 && team_member.league_rank() == 0)
        // printf("%i\n", team_member.team_size());
        // -------------Load into scratch memory----------------------------
        if (team_member.team_rank() == 0) {

          for (int ix = 0; ix < ngllx; ix++) {
            s_wxgll(ix) = wxgll(ix);
          }

          for (int iz = 0; iz < ngllz; iz++) {
            s_wzgll(iz) = wzgll(iz);
          }
          // Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, ngllx),
          //                      [=](const int ix) {
          //                        s_wxgll(ix) = wxgll(ix);
          //                        s_wzgll(ix) = wzgll(ix);
          //                      });
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
              const type_real xixl = sv_xix(iz, ix);
              const type_real xizl = sv_xiz(iz, ix);
              const type_real gammaxl = sv_gammax(iz, ix);
              const type_real gammazl = sv_gammaz(iz, ix);
              const type_real jacobianl = sv_jacobian(iz, ix);
              const type_real mul = sv_mu(iz, ix);
              const type_real lambdaplus2mul = sv_lambdaplus2mu(iz, ix);
              const type_real lambdal = lambdaplus2mul - 2.0 * mul;

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

              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllx),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_hprime_xx(ix, l);
              //       type_real sx = s_tempx(iz, l);
              //       sumx += sx * fac;
              //     },
              //     sum_hprime_x1);

              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllx),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_hprime_xx(ix, l);
              //       type_real sz = s_tempz(iz, l);
              //       sumx += sz * fac;
              //     },
              //     sum_hprime_x3);

              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllz),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_hprime_zz(iz, l);
              //       type_real sx = s_tempx(l, ix);
              //       sumx += sx * fac;
              //     },
              //     sum_hprime_z1);

              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllz),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_hprime_zz(iz, l);
              //       type_real sz = s_tempz(l, ix);
              //       sumx += sz * fac;
              //     },
              //     sum_hprime_z3);

              const type_real duxdxl =
                  xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;
              const type_real duxdzl =
                  xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

              const type_real duzdxl =
                  xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;
              const type_real duzdzl =
                  xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

              const type_real duzdxl_plus_duxdzl = duzdxl + duxdzl;
              if (wave == p_sv) {
                // P_SV case
                type_real sigmaxx = lambdaplus2mul * duxdxl + lambdal * duzdzl;
                s_sigma_xx(iz, ix) = sigmaxx;
                type_real sigmazz = lambdaplus2mul * duzdzl + lambdal * duxdxl;
                s_sigma_zz(iz, ix) = sigmazz;
                type_real sigmaxz = mul * duzdxl_plus_duxdzl;
                s_sigma_xz(iz, ix) = sigmaxz;
              } else {
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
              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllx),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_wxgll(l) * s_hprime_xx(l, ix);
              //       sumx += s_tempx(iz, l) * fac;
              //     },
              //     s_tempx1(iz, ix));
            });

        // Kokkos::parallel_for(
        //     Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
        //       const int ix = xz % ngllz;
        //       const int iz = xz / ngllz;
        //       Kokkos::parallel_reduce(
        //           Kokkos::ThreadVectorRange(team_member, ngllx),
        //           [&](const int l, type_real &sumx) {
        //             type_real fac = s_wxgll(l) * s_hprime_xx(l, ix);
        //             sumx += s_tempz(iz, l) * fac;
        //           },
        //           s_tempz1(iz, ix));
        //     });

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
              // Kokkos::parallel_reduce(
              //     Kokkos::ThreadVectorRange(team_member, ngllz),
              //     [&](const int l, type_real &sumx) {
              //       type_real fac = s_wzgll(l) * s_hprime_zz(l, iz);
              //       sumx += s_tempx(l, ix) * fac;
              //     },
              //     s_tempx3(iz, ix));
            });

        // Kokkos::parallel_for(
        //     Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
        //       const int ix = xz % ngllz;
        //       const int iz = xz / ngllz;
        //       Kokkos::parallel_reduce(
        //           Kokkos::ThreadVectorRange(team_member, ngllz),
        //           [&](const int l, type_real &sumx) {
        //             type_real fac = s_wzgll(l) * s_hprime_zz(l, iz);
        //             sumx += s_tempz(l, ix) * fac;
        //           },
        //           s_tempz3(iz, ix));
        //     });

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
      specfem::DeviceRange(0, nglob), KOKKOS_CLASS_LAMBDA(const int iglob) {
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
      specfem::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(const specfem::DeviceTeam::member_type &team_member) {
        int isource = team_member.league_rank();
        int ispec = ispec_array(isource);
        auto sv_ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

        if (ispec_type(ispec) == elastic) {

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

                if (wave == p_sv) {
                  const type_real accelx =
                      source_array(isource, iz, ix, 0) * stf;
                  const type_real accelz =
                      source_array(isource, iz, ix, 1) * stf;
                  Kokkos::single(Kokkos::PerThread(team_member), [=] {
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), accelz);
                  });
                } else {
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
