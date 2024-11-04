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
      field(assembly.fields.get_simulation_field<WavefieldType>()),
      quadrature_points(quadrature_points) {

  // Check if the receiver element is of type being allocated
  for (int ireceiver = 0; ireceiver < nreceivers; ++ireceiver) {
    const int ispec = h_receiver_kernel_index_mapping(ireceiver);
    if ((assembly.properties.h_element_types(ispec) !=
         medium_type::medium_tag) &&
        (assembly.properties.h_element_property(ispec) !=
         medium_type::property_tag)) {
      throw std::runtime_error("Invalid element detected in kernel");
    }
  }

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
                                                 PropertyTag, qp_type, using_simd>();

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
  constexpr int NGLL = quadrature_points_type::NGLL;
  using ElementFieldType = specfem::element::field<
      NGLL, DimensionType, MediumTag, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true, true, false, using_simd>;

  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, DimensionType, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  if (nreceivers == 0)
    return;

  if (nseismograms == 0)
    return;

  int scratch_size =
      ElementFieldType::shmem_size() + ElementQuadratureType::shmem_size();

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
        ElementFieldType element_field(team_member);
        ElementQuadratureType element_quadrature(team_member);

        // Allocate shared views
        // ----------------------------------------------------------------

        specfem::compute::load_on_device(team_member, quadrature,
                                         element_quadrature);
        specfem::compute::load_on_device(team_member, ispec_l, field,
                                         element_field);

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
              const specfem::point::index<DimensionType> index(ispec_l, iz, ix);
              const auto point_partial_derivatives =
                  [&]() {
                    specfem::point::partial_derivatives<DimensionType, false,
                                                        using_simd>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);
                return point_partial_derivatives;
              }();

              const auto point_properties =
                  [&]() -> specfem::point::properties<DimensionType, MediumTag, PropertyTag, using_simd> {
                specfem::point::properties<DimensionType, MediumTag, PropertyTag, using_simd>
                    point_properties;
                specfem::compute::load_on_device(index, properties,
                                                 point_properties);
                return point_properties;
              }();

              const auto active_field = [&]() ->
                  typename ElementFieldType::ViewType {
                    switch (seismogram_type_l) {
                    case specfem::enums::seismogram::type::displacement:
                      return element_field.displacement;
                      break;
                    case specfem::enums::seismogram::type::velocity:
                      return element_field.velocity;
                      break;
                    case specfem::enums::seismogram::type::acceleration:
                      return element_field.acceleration;
                      break;
                    default:
                      DEVICE_ASSERT(false, "seismogram not supported");
                      return {};
                      break;
                    }
                  }();

              auto sv_receiver_field =
                  Kokkos::subview(receivers.receiver_field, iz, ix, iseis_l,
                                  ireceiver_l, isig_step, Kokkos::ALL);

              receiver.get_field(iz, ix, point_partial_derivatives,
                                 point_properties,
                                 element_quadrature.hprime_gll, active_field,
                                 sv_receiver_field);
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
                receivers.cos_recs(ireceiver_l) * seismogram_components(0) +
                receivers.sin_recs(ireceiver_l) * seismogram_components(1);
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 1) =
                receivers.sin_recs(ireceiver_l) * seismogram_components(0) +
                receivers.cos_recs(ireceiver_l) * seismogram_components(1);
          } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0) =
                receivers.cos_recs(ireceiver_l) * seismogram_components(0) +
                receivers.sin_recs(ireceiver_l) * seismogram_components(1);
            receivers.seismogram(isig_step, iseis_l, ireceiver_l, 0) = 0;
          }
        });

        return;
      });
}

#endif /* _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP */
