#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_TPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
specfem::domain::impl::kernels::source_kernel<WavefieldType, DimensionType, MediumTag,
                                              PropertyTag, qp_type>::
    source_kernel(
        const specfem::compute::assembly &assembly,
        const specfem::kokkos::HostView1d<int> h_source_kernel_index_mapping,
        const specfem::kokkos::HostView1d<int> h_source_mapping,
        const quadrature_point_type quadrature_points)
    : nsources(h_source_kernel_index_mapping.extent(0)),
      h_source_kernel_index_mapping(h_source_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      properties(assembly.properties), sources(assembly.sources),
      quadrature_points(quadrature_points),
      global_index_mapping(assembly.fields.get_simulation_field<WavefieldType>().assembly_index_mapping),
      field(assembly.fields.get_simulation_field<WavefieldType>().template get_field<medium_type>()) {

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::element_kernel::check_properties",
      specfem::kokkos::HostRange(0, nsources),
      KOKKOS_LAMBDA(const int isource) {
        const int ispec = h_source_kernel_index_mapping(isource);
        if ((assembly.properties.h_element_types(ispec) !=
             medium_type::medium_tag) &&
            (assembly.properties.h_element_property(ispec) !=
              medium_type::property_tag)) {
          throw std::runtime_error("Invalid element detected in kernel");
        }
      });

  Kokkos::fence();

  source_kernel_index_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::element_kernel_index_"
      "mapping",
      nsources);

  source_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::source_mapping",
      nsources);

  Kokkos::deep_copy(source_kernel_index_mapping, h_source_kernel_index_mapping);
  Kokkos::deep_copy(source_mapping, h_source_mapping);

  source = specfem::domain::impl::sources::source<DimensionType, MediumTag,
                                                  PropertyTag, quadrature_point_type>();
  return;
}

template <specfem::wavefield::type WavefieldType, specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
void specfem::domain::impl::kernels::source_kernel<WavefieldType,
    DimensionType, MediumTag, PropertyTag,
    qp_type>::compute_source_interaction(const int timestep) const {

  constexpr int components = medium_type::components;

  if (nsources == 0)
    return;

  const auto index_mapping = points.index_mapping;

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_source_interaction",
      specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int isource_l = source_mapping(team_member.league_rank());
        const int ispec_l =
            source_kernel_index_mapping(team_member.league_rank());

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = index_mapping(ispec_l, iz, ix);
              int iglob_l = global_index_mapping(
                  iglob, static_cast<int>(medium_type::medium_tag));

              const type_real stf = sources.stf_array(isource_l, timestep);
              const auto point_properties =
                  properties.load_device_properties<medium_type::medium_tag,
                                                    medium_type::property_tag>(
                      ispec_l, iz, ix);
              specfem::kokkos::array_type<type_real, 2> lagrange_interpolant(
                  sources.source_array(isource_l, 0, iz, ix),
                  sources.source_array(isource_l, 1, iz, ix));

              specfem::kokkos::array_type<type_real, components> acceleration;

              source.compute_interaction(stf, lagrange_interpolant,
                                         point_properties, acceleration);

#ifndef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int i = 0; i < components; i++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&] {
                  Kokkos::atomic_add(&field.field_dot_dot(iglob_l, i),
                                     acceleration[i]);
                });
              }
            });
      });

  Kokkos::fence();
  return;
}

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_TPP
