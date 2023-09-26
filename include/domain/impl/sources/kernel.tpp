#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_TPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <class medium, class qp_type, typename... elemental_properties>
specfem::domain::impl::kernels::
    source_kernel<medium, qp_type, elemental_properties...>::source_kernel(
        const specfem::kokkos::DeviceView3d<int> ibool,
        const specfem::kokkos::DeviceView1d<int> ispec,
        const specfem::kokkos::DeviceView1d<int> isource,
        const specfem::compute::properties &properties,
        const specfem::compute::sources &sources,
        quadrature_point_type quadrature_points,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field_dot_dot)
    : ibool(ibool), ispec(ispec), isource(isource), quadrature_points(quadrature_points),
      stf_array(sources.stf_array), field_dot_dot(field_dot_dot) {

#ifndef NDEBUG
  assert(field_dot_dot.extent(1) == medium::components);
#endif

  const auto source_array = sources.source_array;

  source = specfem::domain::impl::sources::source<dimension, medium, qp_type,
                                                  elemental_properties...>(
      properties, source_array);

  return;
}

template <class medium, class qp_type, typename... elemental_properties>
void specfem::domain::impl::kernels::source_kernel<medium, qp_type,
                                                   elemental_properties...>::
    compute_source_interaction(const type_real timeval) const {

  constexpr int components = medium::components;
  const int nsources = this->ispec.extent(0);

  const auto ibool = this->ibool;

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_source_interaction",
      specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int isource_l = isource(team_member.league_rank());
        const int ispec_l = ispec(team_member.league_rank());

        type_real stf;

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, 1),
            [=](const int &, type_real &lsum) {
              lsum = stf_array(isource_l).compute(timeval);
            },
            stf);

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec_l, iz, ix);

              type_real acceleration[components];

              source.compute_interaction(isource_l, ispec_l, xz, stf, acceleration);

#ifndef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int i = 0; i < components; i++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&] {
                  Kokkos::atomic_add(&field_dot_dot(iglob, i), acceleration[i]);
                });
              }
            });
      });

  Kokkos::fence();
  return;
}

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_TPP
