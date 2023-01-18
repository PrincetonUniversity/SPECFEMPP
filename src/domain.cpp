#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    specfem::compute::sources *sources, quadrature::quadrature *quadx,
    quadrature::quadrature *quadz)
    : field(specfem::HostView2d<type_real>("specfem::Domain::Elastic::field",
                                           nglob, ndim)),
      field_dot(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(specfem::HostView2d<type_real>(
          "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)),
      compute(compute), material_properties(material_properties),
      partial_derivatives(partial_derivatives), sources(sources), quadx(quadx),
      quadz(quadz) {

  const specfem::HostView3d<int> ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  // Initialize views
  Kokkos::parallel_for("specfem::Domain::Elastic::initiaze_views",
                       specfem::HostMDrange<2>({ 0, 0 }, { nglob, ndim }),
                       [=](const int iglob, const int idim) {
                         field(iglob, idim) = 0;
                         field_dot(iglob, idim) = 0;
                         field_dot_dot(iglob, idim) = 0;
                         rmass_inverse(iglob, idim) = 0;
                       });

  Kokkos::fence();
  // Compute the mass matrix
  specfem::HostScatterView2d<type_real> results(rmass_inverse);
  auto wxgll = quadx->get_hw();
  auto wzgll = quadz->get_hw();
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      [=](const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = material_properties->rho(ispec, iz, ix);
        auto access = results.access();
        if (material_properties->ispec_type(ispec) == elastic) {
          access(iglob, 0) += wxgll(ix) * wzgll(iz) * rhol *
                              partial_derivatives->jacobian(ispec, iz, ix);
          access(iglob, 1) += wxgll(ix) * wzgll(iz) * rhol *
                              partial_derivatives->jacobian(ispec, iz, ix);
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);
  Kokkos::fence();

  // invert the mass matrix
  Kokkos::parallel_for("specfem::Domain::Elastic::Invert_mass_matrix",
                       specfem::HostRange(0, nglob), [=](const int iglob) {
                         if (rmass_inverse(iglob, 0) > 0.0) {
                           rmass_inverse(iglob, 0) =
                               1.0 / rmass_inverse(iglob, 0);
                           rmass_inverse(iglob, 1) =
                               1.0 / rmass_inverse(iglob, 1);
                         } else {
                           rmass_inverse(iglob, 0) = 1.0;
                           rmass_inverse(iglob, 1) = 1.0;
                         }
                       });

  this->nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->ispec_type(ispec) == elastic) {
      this->nelem_domain++;
    }
  }

  this->ispec_domain = specfem::HostView1d<int>(
      "specfem::Domain::Elastic::ispec_domain", this->nelem_domain);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->ispec_type(ispec) == elastic) {
      this->ispec_domain(index) = ispec;
      index++;
    }
  }

  return;
};

void specfem::Domain::Elastic::compute_stiffness_interaction() {

  const int ngllx = this->quadx->get_N();
  const int ngllz = this->quadz->get_N();
  const int ngllxz = ngllx * ngllz;
  const specfem::HostView1d<type_real> wxgll = this->quadx->get_hw();
  const specfem::HostView1d<type_real> wzgll = this->quadz->get_hw();
  const specfem::HostView2d<type_real> hprime_xx = this->quadx->get_hhprime();
  const specfem::HostView2d<type_real> hprime_zz = this->quadz->get_hhprime();
  const specfem::HostView3d<int> ibool = this->compute->ibool;

  int scratch_size = specfem::HostScratchView1d<type_real>::shmem_size(ngllx);
  scratch_size += specfem::HostScratchView1d<type_real>::shmem_size(ngllz);
  scratch_size +=
      specfem::HostScratchView2d<type_real>::shmem_size(ngllx, ngllx);
  scratch_size +=
      specfem::HostScratchView2d<type_real>::shmem_size(ngllz, ngllz);
  scratch_size +=
      2 * specfem::HostScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      3 * specfem::HostScratchView2d<type_real>::shmem_size(ngllx, ngllz);
  scratch_size +=
      4 * specfem::HostScratchView2d<type_real>::shmem_size(ngllx, ngllz);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_forces",
      specfem::HostTeam(this->nelem_domain, Kokkos::AUTO, ngllx)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const specfem::HostTeam::member_type &team_member) {
        const int ispec = this->ispec_domain(team_member.league_rank());
        auto ibool = Kokkos::subview(this->compute->ibool, ispec, Kokkos::ALL,
                                     Kokkos::ALL);
        auto xix = Kokkos::subview(this->partial_derivatives->xix, ispec,
                                   Kokkos::ALL, Kokkos::ALL);
        auto xiz = Kokkos::subview(this->partial_derivatives->xiz, ispec,
                                   Kokkos::ALL, Kokkos::ALL);
        auto gammax = Kokkos::subview(this->partial_derivatives->gammax, ispec,
                                      Kokkos::ALL, Kokkos::ALL);
        auto gammaz = Kokkos::subview(this->partial_derivatives->gammaz, ispec,
                                      Kokkos::ALL, Kokkos::ALL);
        auto jacobian = Kokkos::subview(this->partial_derivatives->jacobian,
                                        ispec, Kokkos::ALL, Kokkos::ALL);
        auto mu = Kokkos::subview(this->material_properties->mu, ispec,
                                  Kokkos::ALL, Kokkos::ALL);
        auto lambdaplus2mu =
            Kokkos::subview(this->material_properties->lambdaplus2mu, ispec,
                            Kokkos::ALL, Kokkos::ALL);
        int iglob;

        specfem::HostScratchView1d<type_real> s_wxgll(
            team_member.team_scratch(0), ngllx);
        specfem::HostScratchView1d<type_real> s_wzgll(
            team_member.team_scratch(0), ngllz);
        specfem::HostScratchView2d<type_real> s_hprime_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::HostScratchView2d<type_real> s_hprime_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::HostScratchView2d<type_real> s_tempx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_tempz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_sigma_xx(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_sigma_xz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_sigma_zz(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_tempx1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_tempz1(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_tempx3(
            team_member.team_scratch(0), ngllz, ngllx);
        specfem::HostScratchView2d<type_real> s_tempz3(
            team_member.team_scratch(0), ngllz, ngllx);

        // -------------Load shared memory---------------------------------
        if (team_member.team_rank() == 0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, ngllx),
                               [=](const int ix) {
                                 s_wxgll(ix) = wxgll(ix);
                                 s_wzgll(ix) = wzgll(ix);
                               });
        }

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngllxz),
                             [=](const int xz) {
                               const int ix = xz % ngllz;
                               const int iz = xz / ngllz;
                               int iglob = ibool(iz, ix);
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
              type_real xixl = xix(iz, ix);
              type_real xizl = xiz(iz, ix);
              type_real gammaxl = gammax(iz, ix);
              type_real gammazl = gammaz(iz, ix);
              type_real jacobianl = jacobian(iz, ix);
              type_real mul = mu(iz, ix);
              type_real lambdaplus2mul = lambdaplus2mu(iz, ix);
              type_real lambdal = lambdaplus2mul - 2.0 * mul;

              type_real sum_hprime_x1 = 0;
              type_real sum_hprime_x3 = 0;
              type_real sum_hprime_z1 = 0;
              type_real sum_hprime_z3 = 0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllx),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_hprime_xx(ix, l);
                    type_real sx = s_tempx(iz, l);
                    sumx += sx * fac;
                  },
                  sum_hprime_x1);

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllx),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_hprime_xx(ix, l);
                    type_real sz = s_tempz(iz, l);
                    sumx += sz * fac;
                  },
                  sum_hprime_x3);

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllz),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_hprime_zz(iz, l);
                    type_real sx = s_tempx(l, ix);
                    sumx += sx * fac;
                  },
                  sum_hprime_z1);

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllz),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_hprime_zz(iz, l);
                    type_real sz = s_tempz(l, ix);
                    sumx += sz * fac;
                  },
                  sum_hprime_z3);

              type_real duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;
              type_real duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

              type_real duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;
              type_real duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

              type_real duzdxl_plus_duxdzl = duzdxl + duxdzl;
              if (wave == p_sv) {
                // P_SV case
                s_sigma_xx(iz, ix) = lambdaplus2mul * duxdxl + lambdal * duzdzl;
                s_sigma_zz(iz, ix) = lambdaplus2mul * duzdzl + lambdal * duxdxl;
                s_sigma_xz(iz, ix) = mul * duzdxl_plus_duxdzl;
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
              type_real xixl = xix(iz, ix);
              type_real xizl = xiz(iz, ix);
              type_real jacobianl = jacobian(iz, ix);
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
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllx),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_wxgll(l) * s_hprime_xx(l, ix);
                    sumx += s_tempx(iz, l) * fac;
                  },
                  s_tempx1(iz, ix));
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllx),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_wxgll(l) * s_hprime_xx(l, ix);
                    sumx += s_tempz(iz, l) * fac;
                  },
                  s_tempz1(iz, ix));
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              type_real gammaxl = gammax(iz, ix);
              type_real gammazl = gammaz(iz, ix);
              type_real jacobianl = jacobian(iz, ix);
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
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllz),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_wzgll(l) * s_hprime_zz(l, iz);
                    sumx += s_tempx(l, ix) * fac;
                  },
                  s_tempx3(iz, ix));
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, ngllz),
                  [&](const int l, type_real &sumx) {
                    type_real fac = s_wzgll(l) * s_hprime_zz(l, iz);
                    sumx += s_tempz(l, ix) * fac;
                  },
                  s_tempz3(iz, ix));
            });

        team_member.team_barrier();

        // assembles acceleration array
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
              const int ix = xz % ngllz;
              const int iz = xz / ngllz;
              int iglob = ibool(iz, ix);
              type_real sum_terms1 = -1.0 * (s_wzgll(iz) * s_tempx1(iz, ix)) -
                                     (s_wxgll(ix) * s_tempx3(iz, ix));
              type_real sum_terms3 = -1.0 * (s_wzgll(iz) * s_tempz1(iz, ix)) -
                                     (s_wxgll(ix) * s_tempz3(iz, ix));
              Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), sum_terms1);
              Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), sum_terms3);
            });
      });

  Kokkos::fence();
  return;
}

void specfem::Domain::Elastic::divide_mass_matrix() {

  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::HostRange(0, nglob), [=](const int iglob) {
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

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_source_interaction",
      specfem::HostTeam(nsources, Kokkos::AUTO, ngllx),
      KOKKOS_LAMBDA(const specfem::HostTeam::member_type &team_member) {
        int isource = team_member.league_rank();
        int ispec = sources->ispec_array(isource);
        auto ibool = Kokkos::subview(this->compute->ibool, ispec, Kokkos::ALL,
                                     Kokkos::ALL);

        if (material_properties->ispec_type(ispec) == elastic) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;
                int iglob = ibool(iz, ix);

                type_real stf =
                    this->sources->stf_array(isource).T->compute(timeval);

                if (wave == p_sv) {
                  type_real accelx =
                      this->sources->source_array(isource, iz, ix, 0) * stf;
                  type_real accelz =
                      this->sources->source_array(isource, iz, ix, 1) * stf;
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), accelz);
                } else {
                  type_real accelx =
                      this->sources->source_array(isource, iz, ix, 0) * stf;
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                }
              });
        }
      });

  Kokkos::fence();
  return;
}
