#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/enums.h"
#include "../include/globals.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
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
        const int xz = iz * ngllx + ix;
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, xz);
        auto access = results.access();
        if (ispec_type(ispec) == specfem::elements::elastic) {
          access(iglob, 0) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, xz);
          access(iglob, 1) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, xz);
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

template <int NGLL> void specfem::Domain::Elastic::compute_gradients() {

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
  const int NGLL2 = NGLL * NGLL;
  const type_real NGLL_INV = 1.0 / NGLL;
  const Kokkos::View<type_real **, Kokkos::LayoutLeft,
                     specfem::kokkos::DevMemSpace,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess> >
      field = this->field;
  const Kokkos::View<type_real **, Kokkos::LayoutLeft,
                     specfem::kokkos::DevMemSpace,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess> >
      field_dot_dot = this->field_dot_dot;

  static_assert(NGLL2 == NGLL * NGLL);

  assert(hprime_zz.extent(0) == NGLL);
  assert(hprime_xx.extent(0) == NGLL);
  assert(hprime_zz.extent(1) == NGLL);
  assert(hprime_xx.extent(1) == NGLL);
  assert(xix.extent(1) == NGLL2);
  assert(xiz.extent(1) == NGLL2);
  assert(gammax.extent(1) == NGLL2);
  assert(gammaz.extent(1) == NGLL2);
  assert(ibool.extent(1) == NGLL);
  assert(ibool.extent(2) == NGLL);
  assert(lambdaplus2mu.extent(1) == NGLL2);
  assert(mu.extent(1) == NGLL2);

  int scratch_size =
      12 *
      specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL>::shmem_size();

  scratch_size +=
      specfem::kokkos::StaticDeviceScratchView2d<int, NGLL>::shmem_size();

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, 32, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const int ispec = ispec_domain(team_member.league_rank());

        // Assign scratch views
        // Assign scratch views for views that are required by every thread
        // during summations
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_hprime_xx(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_hprime_zz(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL>
            s_hprimewgll_xx(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL>
            s_hprimewgll_zz(team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<int, NGLL> s_iglob(
            team_member.team_scratch(0));

        // Temporary scratch arrays used in calculation of integrals
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp1(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp2(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp3(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp4(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp5(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp6(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp7(
            team_member.team_scratch(0));
        specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_temp8(
            team_member.team_scratch(0));

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;
              const int iglob = ibool(ispec, iz, ix);
              s_temp1(iz, ix) = field(iglob, 0);
              s_temp2(iz, ix) = field(iglob, 1);
              s_temp3(ix, iz) = field(iglob, 0);
              s_temp4(ix, iz) = field(iglob, 1);
              s_temp5(iz, ix) = 0.0;
              s_temp6(iz, ix) = 0.0;
              s_temp7(ix, iz) = 0.0;
              s_temp8(ix, iz) = 0.0;
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
              s_hprimewgll_xx(ix, iz) = wxgll(iz) * s_hprime_xx(iz, ix);
              s_hprimewgll_zz(ix, iz) = wzgll(iz) * s_hprime_zz(iz, ix);
              s_iglob(iz, ix) = iglob;
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;

              type_real sum_hprime_x1 = 0.0;
              type_real sum_hprime_x3 = 0.0;
              type_real sum_hprime_z1 = 0.0;
              type_real sum_hprime_z3 = 0.0;

#pragma unroll
              for (int l = 0; l < NGLL; l++) {
                sum_hprime_x1 += s_hprime_xx(ix, l) * s_temp1(iz, l);
                sum_hprime_x3 += s_hprime_xx(ix, l) * s_temp2(iz, l);
                sum_hprime_z1 += s_hprime_zz(iz, l) * s_temp3(ix, l);
                sum_hprime_z3 += s_hprime_zz(iz, l) * s_temp4(ix, l);
              }

              // duxdx
              s_temp5(iz, ix) = xix(ispec, xz) * sum_hprime_x1 +
                                gammax(ispec, xz) * sum_hprime_x3;

              // duxdz
              s_temp6(iz, ix) = xiz(ispec, xz) * sum_hprime_x1 +
                                gammaz(ispec, xz) * sum_hprime_x3;

              // duzdx
              s_temp7(iz, ix) = xix(ispec, xz) * sum_hprime_z1 +
                                gammax(ispec, xz) * sum_hprime_z3;

              // duzdz
              s_temp8(iz, ix) = xiz(ispec, xz) * sum_hprime_z1 +
                                gammaz(ispec, xz) * sum_hprime_z3;
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz / NGLL;
              const int ix = xz % NGLL;
              const type_real lambdal =
                  lambdaplus2mu(ispec, xz) - 2.0 * mu(ispec, xz);

              if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
                // P_SV case
                // sigma_xx
                s_temp1(iz, ix) = lambdaplus2mu(ispec, xz) * s_temp5(iz, ix) +
                                  lambdal * s_temp8(iz, ix);

                // sigma_zz
                s_temp2(iz, ix) = lambdaplus2mu(ispec, xz) * s_temp8(iz, ix) +
                                  lambdal * s_temp5(iz, ix);

                // sigma_xz
                s_temp3(iz, ix) =
                    mu(ispec, xz) * (s_temp7(iz, ix) + s_temp6(iz, ix));
              } else if (specfem::globals::simulation_wave ==
                         specfem::wave::sh) {
                // SH-case
                // sigma_xx
                s_temp1(iz, ix) =
                    mu(ispec, xz) *
                    s_temp5(iz, ix); // would be sigma_xy in CPU-version

                // sigma_xz
                s_temp3(iz, ix) = mu(ispec, xz) * s_temp6(iz, ix); // sigma_zy
              }
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;
              s_temp5(iz, ix) =
                  jacobian(ispec, xz) * (s_temp1(iz, ix) * xix(ispec, xz) +
                                         s_temp3(iz, ix) * xiz(ispec, xz));
              s_temp6(iz, ix) =
                  jacobian(ispec, xz) * (s_temp3(iz, ix) * xix(ispec, xz) +
                                         s_temp2(iz, ix) * xiz(ispec, xz));
              s_temp7(ix, iz) =
                  jacobian(ispec, xz) * (s_temp1(iz, ix) * gammax(ispec, xz) +
                                         s_temp3(iz, ix) * gammaz(ispec, xz));
              s_temp8(ix, iz) =
                  jacobian(ispec, xz) * (s_temp3(iz, ix) * gammax(ispec, xz) +
                                         s_temp2(iz, ix) * gammaz(ispec, xz));
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
              const int iz = xz * NGLL_INV;
              const int ix = xz - iz * NGLL;

              type_real tempx1 = 0.0;
              type_real tempz1 = 0.0;
              type_real tempx3 = 0.0;
              type_real tempz3 = 0.0;

#pragma unroll
              for (int l = 0; l < NGLL; l++) {
                tempx1 += s_hprimewgll_xx(ix, l) * s_temp1(iz, l);
                tempz1 += s_hprimewgll_xx(ix, l) * s_temp2(iz, l);
                tempx3 += s_hprimewgll_zz(iz, l) * s_temp3(ix, l);
                tempz3 += s_hprimewgll_zz(iz, l) * s_temp4(ix, l);
              }

              const int iglob = s_iglob(iz, ix);
              const type_real sum_terms1 =
                  -1.0 * (wxgll(iz) * tempx1) - (wxgll(ix) * tempx3);
              const type_real sum_terms3 =
                  -1.0 * (wxgll(iz) * tempz1) - (wxgll(ix) * tempz3);
              // Kokkos::single(Kokkos::PerThread(team_member), [=] {
              Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
              Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
              // });
            });
        //               const type_real wzgll_l = -1.0 * wzgll(iz);
        //               const type_real wxgll_l = -1.0 * wxgll(ix);

        // #pragma unroll
        //               for (int l = 0; l < NGLL; l++) {
        //                 tempx1 += s_hprime_xx(l, ix) * s_tempx(iz, l);
        //                 tempz1 += s_hprime_xx(l, ix) * s_tempz(iz, l);
        //                 tempx3 += s_hprime_zz(l, iz) * s_tempx1(l, ix);
        //                 tempz3 += s_hprime_zz(l, iz) * s_tempz1(l, ix);
        //               }

        //               const int iglob = ibool(ispec, iz, ix);
        //               // Kokkos::single(Kokkos::PerThread(team_member),
        //               // [=] {
        //               Kokkos::atomic_add(&field_dot_dot(iglob, 0),
        //                                  (wzgll_l * tempx1 + wxgll_l *
        //                                  tempx3));
        //               Kokkos::atomic_add(&field_dot_dot(iglob, 1),
        //                                  (wzgll_l * tempz1 + wxgll_l *
        //                                  tempz3));
        // });
        // });
      });

  // Kokkos::fence();

  return;
}

// void specfem::Domain::Elastic::compute_stresses() {

//   const int ngllx = this->quadx->get_N();
//   const int ngllz = this->quadz->get_N();
//   const int nspec = this->compute->ibool.extent(0);
//   const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
//   const auto mu = this->material_properties->mu;

//   Kokkos::parallel_for(
//       "specfem::Domain::Elastic::compute_stresses",
//       specfem::kokkos::DeviceMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx
//       }), KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
//         const type_real lambdal =
//             lambdaplus2mu(ispec, iz, ix) - 2.0 * mu(ispec, iz, ix);

//         if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
//           // P_SV case
//           this->sigma_xx(ispec, iz, ix) =
//               lambdaplus2mu(ispec, iz, ix) * this->duxdx(ispec, iz, ix) +
//               lambdal * this->duzdz(ispec, iz, ix);
//           this->sigma_zz(ispec, iz, ix) =
//               lambdaplus2mu(ispec, iz, ix) * this->duzdz(ispec, iz, ix) +
//               lambdal * this->duxdx(ispec, iz, ix);
//           this->sigma_xz(ispec, iz, ix) =
//               mu(ispec, iz, ix) *
//               (this->duzdx(ispec, iz, ix) + this->duxdz(ispec, iz, ix));
//         } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
//           // SH-case
//           this->sigma_xx(ispec, iz, ix) =
//               mu(ispec, iz, ix) *
//               this->duxdx(ispec, iz, ix); // would be sigma_xy in CPU-version
//           this->sigma_xz(ispec, iz, ix) =
//               mu(ispec, iz, ix) * this->duxdz(ispec, iz, ix); // sigma_zy
//         }
//       });

//   Kokkos::fence();

//   return;
// }

// void specfem::Domain::Elastic::compute_integrals() {

//   const int ngllx = this->quadx->get_N();
//   const int ngllz = this->quadz->get_N();
//   const int ngllxz = ngllx * ngllz;
//   const auto wxgll = this->quadx->get_w();
//   const auto wzgll = this->quadz->get_w();
//   const auto hprime_xx = this->quadx->get_hprime();
//   const auto hprime_zz = this->quadz->get_hprime();
//   const auto xix = this->partial_derivatives->xix;
//   const auto xiz = this->partial_derivatives->xiz;
//   const auto gammax = this->partial_derivatives->gammax;
//   const auto gammaz = this->partial_derivatives->gammaz;
//   const auto jacobian = this->partial_derivatives->jacobian;
//   const auto ibool = this->compute->ibool;

//   int scratch_size =
//       specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
//       ngllx);
//   scratch_size +=
//       specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllz,
//       ngllz);
//   scratch_size +=
//       2 *
//       specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
//       ngllz);
//   scratch_size +=
//       4 *
//       specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
//       ngllz);

//   Kokkos::parallel_for(
//       "specfem::Domain::Elastic::compute_gradients",
//       specfem::kokkos::DeviceTeam(this->nelem_domain, 32, 1)
//           .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         const int ispec = ispec_domain(team_member.league_rank());

//         // Assign scratch views
//         specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx(
//             team_member.team_scratch(0), ngllx, ngllx);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz(
//             team_member.team_scratch(0), ngllz, ngllz);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempx(
//             team_member.team_scratch(0), ngllz, ngllx);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempz(
//             team_member.team_scratch(0), ngllz, ngllx);

//         // temporary views to store integrals
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempx1(
//             team_member.team_scratch(0), ngllz, ngllx);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempz1(
//             team_member.team_scratch(0), ngllz, ngllx);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempx3(
//             team_member.team_scratch(0), ngllz, ngllx);
//         specfem::kokkos::DeviceScratchView2d<type_real> s_tempz3(
//             team_member.team_scratch(0), ngllz, ngllx);

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
//             [=](const int ij) {
//               const int i = ij % ngllx;
//               const int j = ij / ngllx;
//               s_hprimewgll_xx(j, i) = wxgll(j) * hprime_xx(j, i);
//             });

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
//             [=](const int ij) {
//               const int i = ij % ngllz;
//               const int j = ij / ngllz;
//               s_hprimewgll_zz(j, i) = wzgll(j) * hprime_zz(j, i);
//             });

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
//               const int ix = xz % ngllz;
//               const int iz = xz / ngllz;
//               s_tempx(iz, ix) =
//                   jacobian(ispec, iz, ix) *
//                   (this->sigma_xx(ispec, iz, ix) * xixl +
//                    this->sigma_xz(ispec, iz, ix) * xizl);
//               s_tempz(iz, ix) =
//                   jacobian(ispec, iz, ix) *
//                   (this->sigma_xz(ispec, iz, ix) * xixl +
//                    this->sigma_zz(ispec, iz, ix) * xizl);
//             });

//         team_member.team_barrier();

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
//               const int ix = xz % ngllz;
//               const int iz = xz / ngllz;

//               s_tempx1(iz, ix) = 0;
//               s_tempz1(iz, ix) = 0;
//               for (int l = 0; l < ngllx; l++) {
//                 s_tempx1(iz, ix) += s_hprimewgll_xx(l, ix) * s_tempx(iz, l);
//                 s_tempz1(iz, ix) += s_hprimewgll_xx(l, ix) * s_tempz(iz, l);
//               }
//             });

//         team_member.team_barrier();

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
//               const int ix = xz % ngllz;
//               const int iz = xz / ngllz;
//               s_tempx(iz, ix) =
//                   jacobian(ispec, iz, ix) *
//                   (this->sigma_xx(ispec, iz, ix) * gammaxl +
//                    this->sigma_xz(ispec, iz, ix) * gammazl);
//               s_tempz(iz, ix) =
//                   jacobian(ispec, iz, ix) *
//                   (this->sigma_xz(ispec, iz, ix) * gammaxl +
//                    this->sigma_zz(ispec, iz, ix) * gammazl);
//             });

//         team_member.team_barrier();

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
//               const int ix = xz % ngllz;
//               const int iz = xz / ngllz;

//               s_tempx3(iz, ix) = 0;
//               s_tempz3(iz, ix) = 0;
//               for (int l = 0; l < ngllz; l++) {
//                 s_tempx3(iz, ix) += s_hprimewgll_zz(l, iz) * s_tempx(l, ix);
//                 s_tempz3(iz, ix) += s_hprimewgll_zz(l, iz) * s_tempz(l, ix);
//               }
//             });

//         team_member.team_barrier();

//         // assembles acceleration array
//         Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngllxz),
//                              [=](const int xz) {
//                                const int ix = xz % ngllz;
//                                const int iz = xz / ngllz;
//                                const int iglob = ibool(ispec, iz, ix);
//                                const type_real sum_terms1 =
//                                    -1.0 * (wzgll(iz) * s_tempx1(iz, ix)) -
//                                    (wxgll(ix) * s_tempx3(iz, ix));
//                                const type_real sum_terms3 =
//                                    -1.0 * (wzgll(iz) * s_tempz1(iz, ix)) -
//                                    (wxgll(ix) * s_tempz3(iz, ix));
//                                //
//                                Kokkos::single(Kokkos::PerThread(team_member),
//                                // [=] {
//                                //
//                                Kokkos::atomic_add(&this->field_dot_dot(iglob,
//                                //   0), sum_terms1);
//                                //
//                                Kokkos::atomic_add(&this->field_dot_dot(iglob,
//                                //   1), sum_terms3);
//                                // });
//                              });
//       });

//   Kokkos::fence();
//   return;
// }

void specfem::Domain::Elastic::compute_stiffness_interaction() {

  this->compute_gradients<5>();
  // this->compute_stresses();
  // this->compute_integrals();

  // const int ngllx = this->quadx->get_N();
  // const int ngllz = this->quadz->get_N();
  // const int ngllxz = ngllx * ngllz;
  // const auto ibool = this->compute->ibool;
  // const auto ispec_domain = this->ispec_domain;
  // const auto xix = this->partial_derivatives->xix;
  // const auto xiz = this->partial_derivatives->xiz;
  // const auto gammax = this->partial_derivatives->gammax;
  // const auto gammaz = this->partial_derivatives->gammaz;
  // const auto jacobian = this->partial_derivatives->jacobian;
  // const auto mu = this->material_properties->mu;
  // const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
  // const auto wxgll = this->quadx->get_w();
  // const auto wzgll = this->quadz->get_w();
  // const auto hprime_xx = this->quadx->get_hprime();
  // const auto hprime_zz = this->quadz->get_hprime();

  // int scratch_size =
  //     specfem::kokkos::DeviceScratchView1d<type_real>::shmem_size(ngllx);
  // scratch_size +=
  //     specfem::kokkos::DeviceScratchView1d<type_real>::shmem_size(ngllz);
  // scratch_size +=
  //     specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
  //     ngllx);
  // scratch_size +=
  //     specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllz,
  //     ngllz);
  // scratch_size +=
  //     2 *
  //     specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
  //     ngllz);
  // scratch_size +=
  //     3 *
  //     specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
  //     ngllz);
  // scratch_size +=
  //     4 *
  //     specfem::kokkos::DeviceScratchView2d<type_real>::shmem_size(ngllx,
  //     ngllz);

  // Kokkos::parallel_for(
  //     "specfem::Domain::Elastic::compute_forces",
  //     specfem::kokkos::DeviceTeam(this->nelem_domain, 32, 1)
  //         .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
  //     KOKKOS_CLASS_LAMBDA(
  //         const specfem::kokkos::DeviceTeam::member_type &team_member) {
  //       // std::cout << team_member.league_rank() << std::endl;
  //       const int ispec = ispec_domain(team_member.league_rank());

  //       // Getting subviews for better readability
  //       // This has a small perfomance hit (It should be negligible)
  //       const auto sv_ibool =
  //           Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_xix =
  //           Kokkos::subview(xix, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_xiz =
  //           Kokkos::subview(xiz, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_gammax =
  //           Kokkos::subview(gammax, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_gammaz =
  //           Kokkos::subview(gammaz, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_jacobian =
  //           Kokkos::subview(jacobian, ispec, Kokkos::ALL, Kokkos::ALL);
  //       const auto sv_mu = Kokkos::subview(mu, ispec, Kokkos::ALL,
  //       Kokkos::ALL); const auto sv_lambdaplus2mu =
  //           Kokkos::subview(lambdaplus2mu, ispec, Kokkos::ALL, Kokkos::ALL);

  //       // Assign scratch views
  //       // Optional performance related scratch views
  //       specfem::kokkos::DeviceScratchView1d<type_real> s_wxgll(
  //           team_member.team_scratch(0), ngllx);
  //       specfem::kokkos::DeviceScratchView1d<type_real> s_wzgll(
  //           team_member.team_scratch(0), ngllz);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx(
  //           team_member.team_scratch(0), ngllx, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz(
  //           team_member.team_scratch(0), ngllz, ngllz);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempx(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempz(
  //           team_member.team_scratch(0), ngllz, ngllx);

  //       // Nacessary scratch views
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_xx(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_xz(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_sigma_zz(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempx1(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempz1(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempx3(
  //           team_member.team_scratch(0), ngllz, ngllx);
  //       specfem::kokkos::DeviceScratchView2d<type_real> s_tempz3(
  //           team_member.team_scratch(0), ngllz, ngllx);

  //       // -------------Load into scratch memory----------------------------
  //       if (team_member.team_rank() == 0) {

  //         for (int ix = 0; ix < ngllx; ix++) {
  //           s_wxgll(ix) = wxgll(ix);
  //         }

  //         for (int iz = 0; iz < ngllz; iz++) {
  //           s_wzgll(iz) = wzgll(iz);
  //         }
  //       }

  //       Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngllxz),
  //                            [=](const int xz) {
  //                              const int ix = xz % ngllz;
  //                              const int iz = xz / ngllz;
  //                              int iglob = sv_ibool(iz, ix);
  //                              s_tempx(iz, ix) = this->field(iglob, 0);
  //                              s_tempz(iz, ix) = this->field(iglob, 1);
  //                            });

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
  //           [=](const int ij) {
  //             const int i = ij % ngllx;
  //             const int j = ij / ngllx;
  //             s_hprime_xx(j, i) = hprime_xx(j, i);
  //           });

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
  //           [=](const int ij) {
  //             const int i = ij % ngllz;
  //             const int j = ij / ngllz;
  //             s_hprime_zz(j, i) = hprime_zz(j, i);
  //           });
  //       //----------------------------------------------------------------

  //       team_member.team_barrier();

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;

  //             type_real sum_hprime_x1 = 0;
  //             type_real sum_hprime_x3 = 0;
  //             type_real sum_hprime_z1 = 0;
  //             type_real sum_hprime_z3 = 0;

  //             for (int l = 0; l < ngllx; l++) {
  //               sum_hprime_x1 += s_hprime_xx(ix, l) * s_tempx(iz, l);
  //               sum_hprime_x3 += s_hprime_xx(ix, l) * s_tempz(iz, l);
  //             }

  //             for (int l = 0; l < ngllz; l++) {
  //               sum_hprime_z1 += s_hprime_zz(iz, l) * s_tempx(l, ix);
  //               sum_hprime_z3 += s_hprime_zz(iz, l) * s_tempz(l, ix);
  //             }

  //             const type_real xixl = sv_xix(iz, ix);
  //             const type_real xizl = sv_xiz(iz, ix);
  //             const type_real gammaxl = sv_gammax(iz, ix);
  //             const type_real gammazl = sv_gammaz(iz, ix);
  //             const type_real jacobianl = sv_jacobian(iz, ix);

  //             const type_real duxdxl =
  //                 xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;
  //             const type_real duxdzl =
  //                 xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  //             const type_real duzdxl =
  //                 xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;
  //             const type_real duzdzl =
  //                 xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  //             const type_real duzdxl_plus_duxdzl = duzdxl + duxdzl;

  //             const type_real mul = sv_mu(iz, ix);
  //             const type_real lambdaplus2mul = sv_lambdaplus2mu(iz, ix);
  //             const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  //             if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
  //               // P_SV case
  //               s_sigma_xx(iz, ix) = lambdaplus2mul * duxdxl + lambdal *
  //               duzdzl; s_sigma_zz(iz, ix) = lambdaplus2mul * duzdzl +
  //               lambdal * duxdxl; s_sigma_xz(iz, ix) = mul *
  //               duzdxl_plus_duxdzl;
  //             } else if (specfem::globals::simulation_wave ==
  //                        specfem::wave::sh) {
  //               // SH-case
  //               s_sigma_xx(iz, ix) =
  //                   mul * duxdxl; // would be sigma_xy in CPU-version
  //               s_sigma_xz(iz, ix) = mul * duxdzl; // sigma_zy
  //             }
  //           });

  //       team_member.team_barrier();

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;
  //             const type_real xixl = sv_xix(iz, ix);
  //             const type_real xizl = sv_xiz(iz, ix);
  //             const type_real jacobianl = sv_jacobian(iz, ix);
  //             s_tempx(iz, ix) = jacobianl * (s_sigma_xx(iz, ix) * xixl +
  //                                            s_sigma_xz(iz, ix) * xizl);
  //             s_tempz(iz, ix) = jacobianl * (s_sigma_xz(iz, ix) * xixl +
  //                                            s_sigma_zz(iz, ix) * xizl);
  //           });

  //       team_member.team_barrier();

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;

  //             s_tempx1(iz, ix) = 0;
  //             s_tempz1(iz, ix) = 0;
  //             for (int l = 0; l < ngllx; l++) {
  //               s_tempx1(iz, ix) +=
  //                   s_wxgll(l) * s_hprime_xx(l, ix) * s_tempx(iz, l);
  //               s_tempz1(iz, ix) +=
  //                   s_wxgll(l) * s_hprime_xx(l, ix) * s_tempz(iz, l);
  //             }
  //           });

  //       team_member.team_barrier();

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;
  //             const type_real gammaxl = sv_gammax(iz, ix);
  //             const type_real gammazl = sv_gammaz(iz, ix);
  //             const type_real jacobianl = sv_jacobian(iz, ix);
  //             s_tempx(iz, ix) = jacobianl * (s_sigma_xx(iz, ix) * gammaxl +
  //                                            s_sigma_xz(iz, ix) * gammazl);
  //             s_tempz(iz, ix) = jacobianl * (s_sigma_xz(iz, ix) * gammaxl +
  //                                            s_sigma_zz(iz, ix) * gammazl);
  //           });

  //       team_member.team_barrier();

  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [&](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;

  //             s_tempx3(iz, ix) = 0;
  //             s_tempz3(iz, ix) = 0;
  //             for (int l = 0; l < ngllz; l++) {
  //               s_tempx3(iz, ix) +=
  //                   s_wzgll(l) * s_hprime_zz(l, iz) * s_tempx(l, ix);
  //               s_tempz3(iz, ix) +=
  //                   s_wzgll(l) * s_hprime_zz(l, iz) * s_tempz(l, ix);
  //             }
  //           });

  //       team_member.team_barrier();

  //       // assembles acceleration array
  //       Kokkos::parallel_for(
  //           Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
  //             const int ix = xz % ngllz;
  //             const int iz = xz / ngllz;
  //             const int iglob = sv_ibool(iz, ix);
  //             const type_real sum_terms1 =
  //                 -1.0 * (s_wzgll(iz) * s_tempx1(iz, ix)) -
  //                 (s_wxgll(ix) * s_tempx3(iz, ix));
  //             const type_real sum_terms3 =
  //                 -1.0 * (s_wzgll(iz) * s_tempz1(iz, ix)) -
  //                 (s_wxgll(ix) * s_tempz3(iz, ix));
  //             Kokkos::single(Kokkos::PerThread(team_member), [=] {
  //               Kokkos::atomic_add(&this->field_dot_dot(iglob, 0),
  //               sum_terms1); Kokkos::atomic_add(&this->field_dot_dot(iglob,
  //               1), sum_terms3);
  //             });
  //           });
  //     });

  // Kokkos::fence();

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

// Copy the values of relevant field at the spectral element where receiver lies
// to a storage view (field)
// KOKKOS_FUNCTION
// void get_receiver_field_for_seismogram(
//     const specfem::kokkos::DeviceTeam::member_type &team_member,
//     specfem::kokkos::DeviceView3d<type_real> field,
//     const specfem::kokkos::DeviceView2d<int> sv_ibool,
//     specfem::kokkos::DeviceView2d<type_real> copy_field) {

//   const int ngllx = sv_ibool.extent(0);
//   const int ngllz = sv_ibool.extent(1);
//   const int ngllxz = ngllx * ngllz;

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
//         const int ix = xz % ngllz;
//         const int iz = xz / ngllz;
//         const int iglob = sv_ibool(iz, ix);

//         if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
//           field(0, iz, ix) = copy_field(iglob, 0);
//           field(1, iz, ix) = copy_field(iglob, 0);
//         } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
//           field(0, iz, ix) = copy_field(iglob, 0);
//         }
//       });

//   return;
// }

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
