#ifndef _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP
#define _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP

#include "algorithms/interpolate.hpp"
#include "domain/impl/receivers/acoustic/interface.hpp"
#include "domain/impl/receivers/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
specfem::domain::impl::kernels::receiver_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag, qp_type>::
    receiver_kernel(
        const specfem::compute::assembly &assembly,
        const specfem::kokkos::HostView1d<int> h_receiver_kernel_index_mapping,
        const specfem::kokkos::HostView1d<int> h_receiver_mapping,
        const quadrature_points_type &quadrature_points)
    : nreceivers(h_receiver_kernel_index_mapping.extent(0)),
      nseismograms(assembly.receivers.seismogram_types.extent(0)),
      h_receiver_kernel_index_mapping(h_receiver_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      properties(assembly.properties), receivers(assembly.receivers),
      field(assembly.fields.get_simulation_field<WavefieldType>()
                .template get_field<MediumTag>()),
      global_index_mapping(assembly.fields.get_simulation_field<WavefieldType>()
                               .assembly_index_mapping),
      quadrature_points(quadrature_points) {

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::element_kernel::check_properties",
      specfem::kokkos::HostRange(0, nreceivers),
      KOKKOS_LAMBDA(const int isource) {
        const int ispec = h_receiver_kernel_index_mapping(isource);
        if ((assembly.properties.h_element_types(ispec) !=
             medium_type::medium_tag) &&
            (assembly.properties.h_element_property(ispec) !=
             medium_type::property_tag)) {
          throw std::runtime_error("Invalid element detected in kernel");
        }
      });

  Kokkos::fence();

  receiver_kernel_index_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::element_kernel_index_"
      "mapping",
      nreceivers);

  receiver_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::source_mapping",
      nreceivers);

  Kokkos::deep_copy(receiver_kernel_index_mapping,
                    h_receiver_kernel_index_mapping);
  Kokkos::deep_copy(receiver_mapping, h_receiver_mapping);

  receiver =
      specfem::domain::impl::receivers::receiver<DimensionType, MediumTag,
                                                 PropertyTag, qp_type>();

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
void specfem::domain::impl::kernels::receiver_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    qp_type>::compute_seismograms(const int &isig_step) const {

  // Allocate scratch views for field, field_dot & field_dot_dot. Incase of
  // acostic domains when calculating displacement, velocity and acceleration
  // seismograms we need to compute the derivatives of the field variables. This
  // requires summing over all lagrange derivatives at all quadrature points
  // within the element. Scratch views speed up this computation by limiting
  // global memory accesses.

  constexpr int components = medium_type::components;

  if (nreceivers == 0)
    return;

  if (nseismograms == 0)
    return;

  const auto hprime = quadrature.gll.hprime;
  const auto index_mapping = points.index_mapping;

  // hprime_xx
  int scratch_size = quadrature_points.template shmem_size<
      type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

  // field, field_dot, field_dot_dot
  scratch_size +=
      3 *
      quadrature_points
          .template shmem_size<type_real, components, specfem::enums::axes::z,
                               specfem::enums::axes::x>();

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_seismogram",
      specfem::kokkos::DeviceTeam(nreceivers * nseismograms, Kokkos::AUTO, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        // --- Get receiver index, seismogram type, and spectral element index
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto iseis_l = team_member.league_rank() % nseismograms;
        const auto seismogram_type_l = receivers.seismogram_types(iseis_l);
        const int ireceiver_l =
            receiver_mapping(team_member.league_rank() / nseismograms);
        const int ispec_l = receiver_kernel_index_mapping(
            team_member.league_rank() / nseismograms);
        // --------------------------------------------------------------------------

        // Instantiate shared views
        // ----------------------------------------------------------------
        auto s_hprime = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_field =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_field_dot =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_field_dot_dot =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        // Allocate shared views
        // ----------------------------------------------------------------

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime(iz, ix, 0) = hprime(iz, ix);
              const int iglob = global_index_mapping(
                  index_mapping(ispec_l, iz, ix),
                  static_cast<int>(medium_type::medium_tag));
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                const type_real displacement = field.field(iglob, icomponent);
                s_field(iz, ix, icomponent) = field.field(iglob, icomponent);
                const type_real velocity = field.field_dot(iglob, icomponent);
                s_field_dot(iz, ix, icomponent) =
                    field.field_dot(iglob, icomponent);
                const type_real acceleration =
                    field.field_dot_dot(iglob, icomponent);
                s_field_dot_dot(iz, ix, icomponent) =
                    field.field_dot_dot(iglob, icomponent);
              }
            });

        team_member.team_barrier();

        // Get seismogram field
        // ----------------------------------------------------------------

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              const specfem::point::index index(ispec_l, iz, ix);
              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2<false> {
                specfem::point::partial_derivatives2<false>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);
                return point_partial_derivatives;
              }();

              const auto point_properties =
                  [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
                specfem::point::properties<MediumTag, PropertyTag>
                    point_properties;
                specfem::compute::load_on_device(index, properties,
                                                 point_properties);
                return point_properties;
              }();

              const auto active_field = [&]() {
                switch (seismogram_type_l) {
                case specfem::enums::seismogram::type::displacement:
                  return s_field;
                  break;
                case specfem::enums::seismogram::type::velocity:
                  return s_field_dot;
                  break;
                case specfem::enums::seismogram::type::acceleration:
                  return s_field_dot_dot;
                  break;
                default:
                  ASSERT(false, "seismogram not supported");
                  return decltype(s_field){};
                  break;
                }
              }();

              const auto sv_receiver_field =
                  Kokkos::subview(receivers.receiver_field, iz, ix, iseis_l,
                                  ireceiver_l, isig_step, Kokkos::ALL);

              receiver.get_field(iz, ix, point_partial_derivatives,
                                 point_properties, s_hprime, active_field,
                                 sv_receiver_field);
              // receiver.get_field(ireceiver_l, iseis_l, ispec_l,
              //                    seismogram_type_l, xz, isig_step, s_field,
              //                    s_field_dot, s_field_dot_dot);
            });

        team_member.team_barrier();

        // compute seismograms components
        //-------------------------------------------------------------------
        const auto sv_receiver_field =
            Kokkos::subview(receivers.receiver_field, Kokkos::ALL, Kokkos::ALL,
                            iseis_l, ireceiver_l, isig_step, Kokkos::ALL);

        const auto polynomial = Kokkos::subview(
            receivers.receiver_array, ireceiver_l, 0, Kokkos::ALL, Kokkos::ALL);

        const auto seismogram_components =
            specfem::algorithms::interpolate_function(team_member, polynomial,
                                                      sv_receiver_field);

        Kokkos::single(Kokkos::PerTeam(team_member), [=] {
          if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0) =
                receivers.cos_recs(ireceiver_l) * seismogram_components[0] +
                receivers.sin_recs(ireceiver_l) * seismogram_components[1];
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 1) =
                receivers.sin_recs(ireceiver_l) * seismogram_components[0] +
                receivers.cos_recs(ireceiver_l) * seismogram_components[1];
          } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0) =
                receivers.cos_recs(ireceiver_l) * seismogram_components[0] +
                receivers.sin_recs(ireceiver_l) * seismogram_components[1];
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0) = 0;
          }
        });

        specfem::kokkos::array_type<type_real, 2> t_values(
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0),
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 1));

        return;

        // case specfem::enums::seismogram::type::displacement:
        // case specfem::enums::seismogram::type::velocity:
        // case specfem::enums::seismogram::type::acceleration:
        //   specfem::kokkos::array_type<type_real, 2> seismogram_components;
        //   Kokkos::parallel_reduce(
        //       quadrature_points.template
        //       TeamThreadRange<specfem::enums::axes::z,
        //                                                  specfem::enums::axes::x>(
        //           team_member),
        //       [=](const int xz, specfem::kokkos::array_type<type_real, 2>
        //                             &l_seismogram_components) {
        //         receiver.compute_seismogram_components(
        //             ireceiver_l, iseis_l, seismogram_type_l, xz, isig_step,
        //             l_seismogram_components);
        //       },
        //       specfem::kokkos::Sum<specfem::kokkos::array_type<type_real, 2>
        //       >(
        //           seismogram_components));
        //   auto sv_receiver_seismogram = Kokkos::subview(
        //       receiver_seismogram, isig_step, iseis_l, ireceiver_l,
        //       Kokkos::ALL);
        //   Kokkos::single(Kokkos::PerTeam(team_member), [=] {
        //     receiver.compute_seismogram(ireceiver_l, seismogram_components,
        //                                 sv_receiver_seismogram);
        //   });
        //   break;
        //       }
      });
}

#endif /* _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP */
