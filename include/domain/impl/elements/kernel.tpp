#ifndef _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
#define _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/interface.hpp"
#include "domain/impl/elements/elastic/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <class medium, class qp_type, typename... elemental_properties>
specfem::domain::impl::kernels::
    element_kernel<medium, qp_type, elemental_properties...>::element_kernel(
        const specfem::kokkos::DeviceView3d<int> ibool,
        const specfem::kokkos::DeviceView1d<int> ispec,
        const specfem::compute::partial_derivatives &partial_derivatives,
        const specfem::compute::properties &properties,
        quadrature::quadrature *quadx, quadrature::quadrature *quadz,
        qp_type quadrature_points,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field_dot_dot,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            mass_matrix)
    : ibool(ibool), ispec(ispec), quadx(quadx), quadz(quadz),
      quadrature_points(quadrature_points), field(field),
      field_dot_dot(field_dot_dot), mass_matrix(mass_matrix) {

#ifndef NDEBUG
  assert(field.extent(1) == medium::components);
  assert(field_dot_dot.extent(1) == medium::components);
  assert(mass_matrix.extent(1) == medium::components);
#endif

  element = specfem::domain::impl::elements::element<
      dimension, medium_type, quadrature_point_type, elemental_properties...>(
      partial_derivatives, properties);
  return;
}

template <class medium, class qp_type, typename... elemental_properties>
void specfem::domain::impl::kernels::element_kernel<
    medium, qp_type, elemental_properties...>::compute_mass_matrix() const {

  constexpr int components = medium::components;
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::compute_mass_matrix",
      specfem::kokkos::DeviceTeam(ispec.extent(0), Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l = ispec(team_member.league_rank());

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec_l, iz, ix);

              type_real mass_matrix_element[components];

              element.compute_mass_matrix_component(ispec_l, xz,
                                                    mass_matrix_element);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&mass_matrix(iglob, icomponent),
                                     wxgll(ix) * wzgll(iz) *
                                         mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();
  return;
}

template <class medium, class qp_type, typename... elemental_properties>
void specfem::domain::impl::kernels::element_kernel<
    medium, qp_type, elemental_properties...>::compute_stiffness_interaction()
    const {

  constexpr int components = medium::components;
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  const int nelements = ispec.extent(0);

  // s_hprime_xx, s_hprimewgll_xx
  int scratch_size =
      2 * quadrature_points.template shmem_size<
              type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

  // s_hprime_zz, s_hprimewgll_zz
  scratch_size +=
      2 * quadrature_points.template shmem_size<
              type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>();

  // s_field, s_stress_integrand_xi, s_stress_integrand_gamma
  scratch_size +=
      3 *
      quadrature_points
          .template shmem_size<type_real, components, specfem::enums::axes::x,
                               specfem::enums::axes::z>();

  // s_iglob
  scratch_size +=
      quadrature_points.template shmem_size<int, 1, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_interaction",
      specfem::kokkos::DeviceTeam(nelements, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l = ispec(team_member.league_rank());

        // Instantiate shared views
        // ---------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));
        auto s_hprimewgll_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprimewgll_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_field =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_stress_integrand_xi =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_stress_integrand_gamma =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_iglob = quadrature_points.template ScratchView<
            int, 1, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix, 0) = hprime_xx(iz, ix);
              s_hprimewgll_xx(ix, iz, 0) = wxgll(iz) * hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix, 0) = hprime_zz(iz, ix);
              s_hprimewgll_zz(ix, iz, 0) = wzgll(iz) * hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              const int iglob = ibool(ispec_l, iz, ix);
              s_iglob(iz, ix, 0) = iglob;
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_field(iz, ix, icomponent) = field(iglob, icomponent);
                s_stress_integrand_xi(iz, ix, icomponent) = 0.0;
                s_stress_integrand_gamma(iz, ix, icomponent) = 0.0;
              }
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

              type_real dudxl[components];
              type_real dudzl[components];

              type_real stress_integrand_xi[components];
              type_real stress_integrand_gamma[components];

              element.compute_gradient(ispec_l, xz, s_hprime_xx, s_hprime_zz,
                                       s_field, dudxl, dudzl);

              element.compute_stress(ispec_l, xz, dudxl, dudzl,
                                     stress_integrand_xi,
                                     stress_integrand_gamma);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_stress_integrand_xi(iz, ix, icomponent) =
                    stress_integrand_xi[icomponent];
                s_stress_integrand_gamma(iz, ix, icomponent) =
                    stress_integrand_gamma[icomponent];
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);

              const int iglob = s_iglob(iz, ix, 0);
              const type_real wxglll = wxgll(ix);
              const type_real wzglll = wzgll(iz);

              type_real acceleration[components];

              element.compute_acceleration(
                  xz, wxglll, wzglll, s_stress_integrand_xi,
                  s_stress_integrand_gamma, s_hprimewgll_xx, s_hprimewgll_zz,
                  acceleration);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&field_dot_dot(iglob, icomponent),
                                     acceleration[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();

  return;
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
