#ifndef _ELASTIC_DOMAIN_TPP
#define _ELASTIC_DOMAIN_TPP

#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/domain.hpp"
#include "domain/elastic/elastic_domain.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "mathematical_operators/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

// // Specialized kernel when NGLLX == NGLLZ
// // This kernel is templated for compiler optimizations.
// // Specific instances of this kernel should be instantiated inside the kernel
// // calling routine
// template <int NGLL>
// void specfem::domain::domain<specfem::enums::element::medium::elastic,
//                              NGLL>::compute_stiffness_interaction() {

//   const auto hprime_xx = this->quadx->get_hprime();
//   const auto hprime_zz = this->quadz->get_hprime();
//   const auto wxgll = this->quadx->get_w();
//   const auto wzgll = this->quadz->get_w();
//   const auto ispec_domain = this->ispec_domain;
//   const auto field = this->field;

//   constexpr int NGLL2 = NGLL * NGLL;

//   int scratch_size =
//       10 * specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL,
//                                                       NGLL>::shmem_size();

//   scratch_size +=
//       specfem::kokkos::StaticDeviceScratchView2d<int, NGLL,
//       NGLL>::shmem_size();

//   Kokkos::parallel_for(
//       "specfem::Domain::Elastic::compute_gradients",
//       specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
//           .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         const int ispec = ispec_domain(team_member.league_rank());
//         const auto element = this->elements(team_member.league_rank());

//         // Assign scratch views
//         // Assign scratch views for views that are required by every thread
//         // during summations
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_hprime_xx(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_hprime_zz(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_hprimewgll_xx(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_hprimewgll_zz(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<int, NGLL, NGLL> s_iglob(
//             team_member.team_scratch(0));

//         // Temporary scratch arrays used in calculation of integrals
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_fieldx(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_fieldz(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_temp1(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_temp2(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_temp3(team_member.team_scratch(0));
//         specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
//             s_temp4(team_member.team_scratch(0));

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
//               const int iz = xz * NGLL_INV;
//               const int ix = xz - iz * NGLL;
//               const int iglob = ibool(ispec, iz, ix);
//               s_fieldx(iz, ix) = field(iglob, 0);
//               s_fieldz(iz, ix) = field(iglob, 1);
//               s_temp1(iz, ix) = 0.0;
//               s_temp2(iz, ix) = 0.0;
//               s_temp3(iz, ix) = 0.0;
//               s_temp4(iz, ix) = 0.0;
//               s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
//               s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
//               s_hprimewgll_xx(ix, iz) = wxgll(iz) * hprime_xx(iz, ix);
//               s_hprimewgll_zz(ix, iz) = wzgll(iz) * hprime_zz(iz, ix);
//               s_iglob(iz, ix) = iglob;
//             });

//         team_member.team_barrier();

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
//               int iz, ix;
//               sub2ind(xz, NGLL, iz, ix);
//               type_real duxdxl, duxdzl, duzdxl, duzdzl;

//               element->compute_gradients(xz, ispec, s_hprime_xx, s_hprime_zz,
//                                          s_fieldx, s_fieldz, duxdxl, duxdzl,
//                                          duzdxl, duzdzl);

//               element->compute_stresses(
//                   xz, ispec, duxdxl, duxdzl, duzdxl, duzdzl, s_temp1(iz, ix),
//                   s_temp2(iz, ix), s_temp3(iz, ix), s_temp4(iz, ix));
//             });

//         team_member.team_barrier();

//         Kokkos::parallel_for(
//             Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
//               int iz, ix;
//               sub2ind(xz, NGLL, iz, ix);

//               const int iglob = s_iglob(iz, ix);
//               const type_real wxglll = wxgll(ix);
//               const type_real wzglll = wzgll(iz);

//               element->update_accel(xz, ispec, iglob, wxglll, wzglll,
//               s_temp1,
//                                     s_temp2, s_temp3, s_temp4,
//                                     s_hprimewgll_xx, s_hprimewgll_zz);
//             });
//       });

//   Kokkos::fence();

//   return;
// }

template <typename element_type>
using container = typename specfem::domain::impl::elements::container<element_type>;

template <class qp_type, class... traits>
using element_type = specfem::domain::impl::elements::element<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic, qp_type, traits...>;

template <class qp_type>
void instantiate_element(
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::properties properties,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::HostMirror1d<int> h_ispec_domain,
    specfem::kokkos::HostMirror1d<container<element_type<qp_type> > >
        h_elements,
    specfem::kokkos::DeviceView1d<container<element_type<qp_type> > >
        elements) {

  for (int i = 0; i < h_ispec_domain.extent(0); i++) {
    element_type<qp_type> *element;

    element = (element_type<qp_type> *)Kokkos::kokkos_malloc<
        specfem::kokkos::DevMemSpace>(sizeof(
        element_type<qp_type, specfem::enums::element::property::isotropic>));

    const int ispec = h_ispec_domain(i);

    Kokkos::parallel_for(
        "specfem::sources::moment_tensor::moment_tensor::allocate_stf",
        specfem::kokkos::DeviceRange(0, 1), KOKKOS_LAMBDA(const int &) {
          new (element)
              element_type<qp_type,
                           specfem::enums::element::property::isotropic>(
                  ispec, partial_derivatives, properties, field_dot_dot);
        });

    Kokkos::fence();

    container<element_type<qp_type> > element_container(element);

    h_elements(i) = element_container;
  }

  Kokkos::deep_copy(elements, h_elements);
}

template <class qp_type>
void assign_elemental_properties(
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::properties properties,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::HostMirror1d<container<element_type<qp_type> > >
        h_elements,
    specfem::kokkos::DeviceView1d<container<element_type<qp_type> > > elements,
    const int &nspec, const int &ngllz, const int &ngllx, int &nelem_domain) {

  // Assign elemental properties
  // ----------------------------------------------------------------------
  nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) ==
        specfem::enums::element::medium::elastic::value) {
      nelem_domain++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_domain(
      "specfem::domain::elastic::h_ispec_domain", nelem_domain);
  specfem::kokkos::HostMirror1d<int> h_ispec_domain =
      Kokkos::create_mirror_view(ispec_domain);

  elements = specfem::kokkos::DeviceView1d<container<element_type<qp_type> > >(
      "specfem::domain::elastic::elements", nelem_domain);
  h_elements = Kokkos::create_mirror_view(elements);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == specfem::enums::element::elastic) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  instantiate_element(partial_derivatives, properties, field_dot_dot,
                      h_ispec_domain, h_elements, elements);
};

template <class qp_type>
specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>::
    domain(const int ndim, const int nglob, const qp_type &quadrature_points,
           specfem::compute::compute *compute,
           specfem::compute::properties material_properties,
           specfem::compute::partial_derivatives partial_derivatives,
           specfem::compute::sources *sources,
           specfem::compute::receivers *receivers,
           specfem::quadrature::quadrature *quadx,
           specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field", nglob,
          specfem::enums::element::medium::elastic::components)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::Domain::Elastic::field_dot", nglob,
          specfem::enums::element::medium::elastic::components)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::field_dot_dot", nglob,
              specfem::enums::element::medium::elastic::components)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::Domain::Elastic::rmass_inverse", nglob,
              specfem::enums::element::medium::elastic::components)),
      quadrature_points(quadrature_points), compute(compute), sources(sources),
      receivers(receivers), quadx(quadx), quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  // Inverse of mass matrix
  //----------------------------------------------------------------------------
  // Initialize views
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 },
          { nglob, specfem::enums::element::medium::elastic::components }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        this->field(iglob, idim) = 0;
        this->field_dot(iglob, idim) = 0;
        this->field_dot_dot(iglob, idim) = 0;
        this->rmass_inverse(iglob, idim) = 0;
      });

  // Compute the mass matrix
  specfem::kokkos::DeviceScatterView2d<type_real, Kokkos::LayoutLeft> results(
      rmass_inverse);
  constexpr int components =
      specfem::enums::element::medium::elastic::components;
  auto wxgll = quadx->get_w();
  auto wzgll = quadz->get_w();
  auto rho = material_properties.rho;
  auto ispec_type = material_properties.ispec_type;
  auto jacobian = partial_derivatives.jacobian;
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::kokkos::DeviceMDrange<3, Kokkos::Iterate::Left>(
          { 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, iz, ix);
        auto access = results.access();
        if (ispec_type(ispec) ==
            specfem::enums::element::medium::elastic::value) {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            access(iglob, icomponent) +=
                wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
          }
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);

  // invert the mass matrix
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::Invert_mass_matrix",
      specfem::kokkos::DeviceRange(0, nglob),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        if (rmass_inverse(iglob, 0) > 0.0) {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            rmass_inverse(iglob, icomponent) =
                1.0 / rmass_inverse(iglob, icomponent);
          }
        } else {
          for (int icomponent = 0; icomponent < components; icomponent++) {
            rmass_inverse(iglob, icomponent) = 1.0;
          }
        }
      });
  // ----------------------------------------------------------------------------

  assign_elemental_properties(partial_derivatives, material_properties,
                              this->field_dot_dot, this->h_elements,
                              this->elements, nspec, ngllz, ngllx,
                              this->nelem_domain);

  return;
};

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field_dot(specfem::sync::kind
                                                          kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_field_dot_dot(specfem::sync::kind
                                                              kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::sync_rmass_inverse(specfem::sync::kind
                                                              kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::divide_mass_matrix() {

  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::kokkos::DeviceRange(
          0, specfem::enums::element::medium::elastic::components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        this->field_dot_dot(iglob, idim) =
            this->field_dot_dot(iglob, idim) * this->rmass_inverse(iglob, idim);
      });

  // Kokkos::fence();

  return;
}

// --------------- kernels -----------------------------------

template <class qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::compute_stiffness_interaction() {

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto field = this->field;

  int scratch_size =
      2 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::x,
                                            specfem::enums::axes::x>();

  scratch_size +=
      2 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::z,
                                            specfem::enums::axes::z>();

  scratch_size +=
      6 *
      quadrature_points.template shmem_size<type_real, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  scratch_size +=
      quadrature_points.template shmem_size<int, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        const auto [ngllx, ngllz] = quadrature_points.get_ngll();
        const auto element = this->elements(team_member.league_rank());
        const auto ispec = element->get_ispec();

        // Instantiate shared views
        // ---------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));
        auto s_hprimewgll_xx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprimewgll_zz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_fieldx = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_fieldz = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_1 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_2 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_3 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_stress_integral_4 = quadrature_points.template ScratchView<
            type_real, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_iglob =
            quadrature_points.template ScratchView<int, specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix) = hprime_xx(iz, ix);
              s_hprimewgll_xx(ix, iz) = wxgll(iz) * hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix) = hprime_zz(iz, ix);
              s_hprimewgll_zz(ix, iz) = wzgll(iz) * hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              const int iglob = ibool(ispec, iz, ix);
              s_fieldx(iz, ix) = field(iglob, 0);
              s_fieldz(iz, ix) = field(iglob, 1);
              s_stress_integral_1(iz, ix) = 0.0;
              s_stress_integral_2(iz, ix) = 0.0;
              s_stress_integral_3(iz, ix) = 0.0;
              s_stress_integral_4(iz, ix) = 0.0;
              s_iglob(iz, ix) = iglob;
            });

        // ------------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              type_real duxdxl, duxdzl, duzdxl, duzdzl;

              element->compute_gradients(xz, ispec, s_hprime_xx, s_hprime_zz,
                                         s_fieldx, s_fieldz, duxdxl, duxdzl,
                                         duzdxl, duzdzl);

              element->compute_stresses(
                  xz, ispec, ngllx, duxdxl, duxdzl, duzdxl, duzdzl,
                  s_stress_integral_1(iz, ix), s_stress_integral_2(iz, ix),
                  s_stress_integral_3(iz, ix), s_stress_integral_4(iz, ix));
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);

              const int iglob = s_iglob(iz, ix);
              const type_real wxglll = wxgll(ix);
              const type_real wzglll = wzgll(iz);

              element->update_accel(xz, ispec, iglob, wxglll, wzglll,
                                    s_stress_integral_1, s_stress_integral_2,
                                    s_stress_integral_3, s_stress_integral_4,
                                    s_hprimewgll_xx, s_hprimewgll_zz);
            });
      });

  Kokkos::fence();

  return;
}

template <typename qp_type>
void specfem::domain::domain<
    specfem::enums::element::medium::elastic,
    qp_type>::compute_source_interaction(const type_real timeval) {

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

        if (ispec_type(ispec) == specfem::enums::element::elastic) {

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

template <typename qp_type>
void specfem::domain::domain<specfem::enums::element::medium::elastic,
                             qp_type>::compute_seismogram(const int isig_step) {

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
        if (ispec_type(ispec) == specfem::enums::element::elastic) {

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

#endif
