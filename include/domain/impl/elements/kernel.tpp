#ifndef _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
#define _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/interface.hpp"
#include "domain/impl/elements/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property,
    specfem::element::boundary_tag boundary, typename quadrature_points_type>
specfem::domain::impl::kernels::element_kernel<
    dimension, medium, property, boundary, quadrature_points_type>::
    element_kernel(
        const specfem::compute::assembly &assembly,
        const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
        const quadrature_point_type &quadrature_points)
    : nelements(h_element_kernel_index_mapping.extent(0)),
      h_element_kernel_index_mapping(h_element_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      properties(assembly.properties),
      boundary_conditions(assembly.boundaries.boundary_types),
      quadrature_points(quadrature_points),
      global_index_mapping(assembly.fields.forward.assembly_index_mapping),
      field(assembly.fields.forward.get_field<medium>()),
      element(assembly, quadrature_points) {

  // Check if the elements being allocated to this kernel are of the correct
  // type
  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::element_kernel::check_properties",
      specfem::kokkos::HostRange(0, nelements),
      KOKKOS_LAMBDA(const int ielement) {
        const int ispec = h_element_kernel_index_mapping(ielement);
        if ((assembly.properties.h_element_types(ispec) !=
             medium_type::value) &&
            (assembly.properties.h_element_property(ispec) !=
             property::value)) {
          throw std::runtime_error("Invalid element detected in kernel");
        }
      });

  Kokkos::fence();

  element_kernel_index_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::element_kernel_index_"
      "mapping",
      nelements);

  Kokkos::deep_copy(element_kernel_index_mapping,
                    h_element_kernel_index_mapping);
  return;
}

template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property,
    specfem::element::boundary_tag boundary, typename quadrature_points_type>
specfem::domain::impl::kernels::element_kernel<
    dimension, medium, property, boundary,
    quadrature_points_type>::compute_mass_matrix() const {
  constexpr int components = medium::components;

  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::compute_mass_matrix",
      specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = index_mapping(ispec_l, iz, ix);
              int iglob_l =
                  global_index_mapping(iglob, static_cast<int>(medium::value));

              const auto point_property = properties.load_device_properties<
                  medium_type::value, property_type::value>(ispec_l, iz, ix);

              const auto point_partial_derivatives =
                  partial_derivatives.load_device_derivatives<true>(ispec_l, iz,
                                                                    ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  mass_matrix_element;

              element.compute_mass_matrix_component(point_property,
                                                    point_partial_derivatives,
                                                    mass_matrix_element);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&field.mass_inverse(iglob_l, icomponent),
                                     wgll(ix) * wgll(iz) *
                                         mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();

  return;
}

template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property,
    specfem::element::boundary_tag boundary, typename quadrature_points_type>
template <specfem::time_scheme::type time_scheme>
specfem::domain::impl::kernels::element_kernel<
    dimension, medium, property, boundary,
    quadrature_points_type>::mass_time_contribution(const type_real dt) const {

  constexpr int components = medium::components;

  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::add_mass_matrix_contribution",
      specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        const auto point_boundary_type = boundary_conditions(ispec_l);

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = index_mapping(ispec_l, iz, ix);

              int iglob_l =
                  global_index_mapping(iglob, static_cast<int>(medium::value));

              const auto point_property = properties.load_device_properties<
                  medium_type::value, property_type::value>(ispec_l, iz, ix);

              const auto point_partial_derivatives =
                  partial_derivatives.load_device_derivatives<true>(ispec_l, iz,
                                                                    ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  mass_matrix_element;

              specfem::kokkos::array_type<type_real, dimension::dim> weight(
                  wgll(ix), wgll(iz));

              element.template mass_time_contribution<time_scheme>(
                  xz, dt, weight, point_partial_derivatives, point_property,
                  point_boundary_type, mass_matrix_element);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&field.mass_inverse(iglob_l, icomponent),
                                     mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();
  return;
}

template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property,
    specfem::element::boundary_tag boundary, typename quadrature_points_type>
specfem::domain::impl::kernels::element_kernel<
    dimension, medium, property, boundary,
    quadrature_points_type>::compute_stiffness_interaction() const {

  constexpr int components = medium::components;

  if (nelements == 0)
    return;

  const auto hprime = quadrature.gll.hprime;
  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  // s_hprime, s_hprimewgll
  int scratch_size =
      2 * quadrature_points.template shmem_size<
              type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

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
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        const auto point_boundary_type = boundary_conditions(ispec_l);

        // Instantiate shared views
        // ---------------------------------------------------------------
        auto s_hprime = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprimewgll = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
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
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime(iz, ix, 0) = hprime(iz, ix);
              s_hprimewgll(ix, iz, 0) = wgll(iz) * hprime(iz, ix);
              const int iglob =
                  global_index_mapping(index_mapping(ispec_l, iz, ix),
                                       static_cast<int>(medium::value));
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                s_field(iz, ix, icomponent) = field.field(iglob, icomponent);
                s_stress_integrand_xi(iz, ix, icomponent) = 0.0;
                s_stress_integrand_gamma(iz, ix, icomponent) = 0.0;
              }
            });

        // ---------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  dudxl;
              specfem::kokkos::array_type<type_real, medium_type::components>
                  dudzl;

              const auto point_partial_derivatives =
                  partial_derivatives.load_device_derivatives<true>(ispec_l, iz,
                                                                    ix);

              element.compute_gradient(xz, s_hprime, s_field,
                                       point_partial_derivatives,
                                       point_boundary_type, dudxl, dudzl);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  stress_integrand_xi;
              specfem::kokkos::array_type<type_real, medium_type::components>
                  stress_integrand_gamma;

              const auto point_property = properties.load_device_properties<
                  medium_type::value, property_type::value>(ispec_l, iz, ix);

              element.compute_stress(xz, dudxl, dudzl,
                                     point_partial_derivatives, point_property,
                                     point_boundary_type, stress_integrand_xi,
                                     stress_integrand_gamma);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
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
              constexpr auto tag = BC::value;

              const int iglob =
                  global_index_mapping(index_mapping(ispec_l, iz, ix),
                                       static_cast<int>(medium::value));
              const specfem::kokkos::array_type<type_real, dimension::dim>
                  weight(wgll(ix), wgll(iz));

              specfem::kokkos::array_type<type_real, medium_type::components>
                  acceleration;

              // Get velocity, partial derivatives, and properties
              // only if needed by the boundary condition
              // ---------------------------------------------------------------
              constexpr bool flag =
                  ((tag == specfem::enums::element::boundary_tag::stacey) ||
                   (tag == specfem::enums::element::boundary_tag::
                               composite_stacey_dirichlet));

              const auto velocity =
                  [&]() -> specfem::kokkos::array_type<type_real, components> {
                if constexpr (flag) {
                  const auto velocity_l =
                      Kokkos::subview(field.field_dot, iglob, Kokkos::ALL);
                  return specfem::kokkos::array_type<type_real, components>(
                      velocity_l);
                } else {
                  return specfem::kokkos::array_type<type_real, components>();
                }
              }();

              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2 {
                if constexpr (flag) {
                  return partial_derivatives.load_device_derivatives<true>(
                      ispec_l, iz, ix);
                } else {
                  return specfem::point::partial_derivatives2();
                }
              }();

              const auto point_property =
                  [&]() -> specfem::point::properties<medium_type::value,
                                                      property_type::value> {
                if constexpr (flag) {
                  return properties.load_device_properties<
                      medium_type::value, property_type::value>(ispec_l, iz,
                                                                ix);
                } else {
                  return specfem::point::properties<medium_type::value,
                                                    property_type::value>();
                }
              }();
              // ---------------------------------------------------------------

              element.compute_acceleration(
                  xz, weight, s_stress_integrand_xi, s_stress_integrand_gamma,
                  s_hprimewgll, point_partial_derivatives, point_property,
                  point_boundary_type, velocity, acceleration);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&field.field_dot_dot(iglob, icomponent),
                                     acceleration[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();

  return;
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
