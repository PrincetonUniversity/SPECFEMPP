#ifndef _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP
#define _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP

#include "domain/impl/receivers/acoustic/interface.hpp"
#include "domain/impl/receivers/elastic/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"

template <class medium, class qp_type, typename... elemental_properties>
specfem::domain::impl::kernels::
    receiver_kernel<medium, qp_type, elemental_properties...>::receiver_kernel(
        const specfem::kokkos::DeviceView3d<int> ibool,
        const specfem::kokkos::DeviceView1d<int> ispec,
        const specfem::kokkos::DeviceView1d<int> ireceiver,
        const specfem::compute::partial_derivatives &partial_derivatives,
        const specfem::compute::properties &properties,
        const specfem::compute::receivers &receivers,
        const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field,
        const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field_dot,
        const specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field_dot_dot,
        specfem::quadrature::quadrature *quadx,
        specfem::quadrature::quadrature *quadz,
        quadrature_points_type quadrature_points)
    : ibool(ibool), ispec(ispec), ireceiver(ireceiver),
      quadrature_points(quadrature_points), field(field), field_dot(field_dot),
      field_dot_dot(field_dot_dot),
      seismogram_types(receivers.seismogram_types), quadx(quadx), quadz(quadz),
      receiver_seismogram(receivers.seismogram) {

#ifndef NDEBUG
  assert(field.extent(1) == medium::components);
  assert(field_dot.extent(1) == medium::components);
  assert(field_dot_dot.extent(1) == medium::components);
#endif

  const auto sin_rec = receivers.sin_recs;
  const auto cos_rec = receivers.cos_recs;
  const auto receiver_array = receivers.receiver_array;
  const auto receiver_field = receivers.receiver_field;

  // Allocate receivers
  this->receiver = specfem::domain::impl::receivers::receiver<
      dimension, medium_type, quadrature_points_type, elemental_properties...>(
      sin_rec, cos_rec, receiver_array, partial_derivatives, properties,
      receiver_field);

  return;
}

template <class medium, class qp_type, typename... elemental_properties>
void specfem::domain::impl::kernels::receiver_kernel<
    medium, qp_type,
    elemental_properties...>::compute_seismograms(const int &isig_step) const {

  // Allocate scratch views for field, field_dot & field_dot_dot. Incase of
  // acostic domains when calculating displacement, velocity and acceleration
  // seismograms we need to compute the derivatives of the field variables. This
  // requires summing over all lagrange derivatives at all quadrature points
  // within the element. Scratch views speed up this computation by limiting
  // global memory accesses.

  constexpr int components = medium::components;
  const int nreceivers = ispec.extent(0);

  if (nreceivers == 0)
    return;

  const int nseismograms = seismogram_types.extent(0);
  const auto ibool = this->ibool;
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  // hprime_xx
  int scratch_size = quadrature_points.template shmem_size<
      type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

  // hprime_zz
  scratch_size += quadrature_points.template shmem_size<
      type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>();

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
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const int ireceiver_l =
            this->ireceiver(team_member.league_rank() / nseismograms);
        const int ispec_l =
            this->ispec(team_member.league_rank() / nseismograms);
        const int iseis_l = team_member.league_rank() % nseismograms;
        const auto seismogram_type_l = this->seismogram_types(iseis_l);

        // Instantiate shared views
        // ----------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
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
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix, 0) = hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix, 0) = hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec_l, iz, ix);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_field(iz, ix, icomponent) = field(iglob, icomponent);
                s_field_dot(iz, ix, icomponent) = field_dot(iglob, icomponent);
                s_field_dot_dot(iz, ix, icomponent) =
                    field_dot_dot(iglob, icomponent);
              }
            });

        // Get seismogram field
        // ----------------------------------------------------------------

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              receiver.get_field(ireceiver_l, iseis_l, ispec_l,
                                 seismogram_type_l, xz, isig_step, s_field,
                                 s_field_dot, s_field_dot_dot, s_hprime_xx,
                                 s_hprime_zz);
            });

        // compute seismograms components
        //-------------------------------------------------------------------
        switch (seismogram_type_l) {
        case specfem::enums::seismogram::type::displacement:
        case specfem::enums::seismogram::type::velocity:
        case specfem::enums::seismogram::type::acceleration:
          dimension::array_type<type_real> seismogram_components;
          Kokkos::parallel_reduce(
              quadrature_points.template TeamThreadRange<
                  specfem::enums::axes::z, specfem::enums::axes::x>(
                  team_member),
              [=](const int xz,
                  dimension::array_type<type_real> &l_seismogram_components) {
                receiver.compute_seismogram_components(
                    ireceiver_l, iseis_l, seismogram_type_l, xz, isig_step,
                    l_seismogram_components);
              },
              specfem::kokkos::Sum<dimension::array_type<type_real> >(
                  seismogram_components));
          auto sv_receiver_seismogram =
              Kokkos::subview(receiver_seismogram, isig_step, iseis_l,
                              ireceiver_l, Kokkos::ALL);
          Kokkos::single(Kokkos::PerTeam(team_member), [=] {
            receiver.compute_seismogram(ireceiver_l, seismogram_components,
                                        sv_receiver_seismogram);
          });
          break;
        }
      });
}

#endif /* _DOMAIN_IMPL_RECEIVERS_KERNEL_TPP */
