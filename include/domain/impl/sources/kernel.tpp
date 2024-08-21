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
specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    qp_type>::source_kernel(const specfem::compute::assembly &assembly,
                            const specfem::kokkos::HostView1d<int>
                                h_source_domain_index_mapping,
                            const quadrature_point_type quadrature_points)
    : nsources(h_source_domain_index_mapping.extent(0)),
      h_source_domain_index_mapping(h_source_domain_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      properties(assembly.properties),
      sources(assembly.sources.get_source_medium<MediumTag>()),
      quadrature_points(quadrature_points),
      field(assembly.fields.get_simulation_field<WavefieldType>()) {

  // Check if the source element is the type being allocated
  for (int isource = 0; isource < nsources; isource++) {
    const int ispec = sources.h_source_index_mapping(isource);
    if ((assembly.properties.h_element_types(ispec) !=
         medium_type::medium_tag) &&
        (assembly.properties.h_element_property(ispec) !=
         medium_type::property_tag)) {
      throw std::runtime_error("Invalid element detected in kernel");
    }
  }

  Kokkos::fence();

  source_domain_index_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::domain::impl::kernels::element_kernel::element_kernel_index_"
      "mapping",
      nsources);

  Kokkos::deep_copy(source_domain_index_mapping, h_source_domain_index_mapping);

  source = specfem::domain::impl::sources::source<
      DimensionType, MediumTag, PropertyTag, quadrature_point_type, using_simd>();
  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, typename qp_type>
void specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    qp_type>::compute_source_interaction(const int timestep) const {

  constexpr int components = medium_type::components;
  using PointFieldType = specfem::point::field<DimensionType, MediumTag, false,
                                               false, true, false, using_simd>;

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
        const int isource_l =
            source_domain_index_mapping(team_member.league_rank());
        const int ispec_l = sources.source_index_mapping(isource_l);

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              specfem::point::index<DimensionType> index(ispec_l, iz, ix);

              const specfem::datatype::ScalarPointViewType<
                  type_real, medium_type::components, using_simd>
                  lagrange_interpolant(Kokkos::subview(
                      sources.source_array, isource_l, Kokkos::ALL, iz, ix));

              // Source time function
              // For acoustic medium, forward simulation, divide by kappa
              const auto stf = [&, timestep, components]() {
                if constexpr ((WavefieldType ==
                               specfem::wavefield::type::forward) &&
                              (MediumTag ==
                               specfem::element::medium_tag::acoustic)) {
                  const auto point_properties = [&]()
                      -> specfem::point::properties<DimensionType, MediumTag,
                                                    PropertyTag, using_simd> {
                    specfem::point::properties<DimensionType, MediumTag,
                                               PropertyTag, using_simd>
                        point_properties;
                    specfem::compute::load_on_device(index, properties,
                                                     point_properties);
                    return point_properties;
                  }();
                  specfem::datatype::ScalarPointViewType<
                      type_real, medium_type::components, using_simd>
                      stf(Kokkos::subview(sources.source_time_function,
                                          timestep, isource_l, Kokkos::ALL));
                  for (int i = 0; i < components; i++) {
                    stf(i) = stf(i) / point_properties.kappa;
                  }
                  return stf;
                } else {
                  return specfem::datatype::ScalarPointViewType<
                      type_real, medium_type::components, using_simd>(
                      Kokkos::subview(sources.source_time_function, timestep,
                                      isource_l, Kokkos::ALL));
                }
              }();

              PointFieldType acceleration;

              source.compute_interaction(stf, lagrange_interpolant,
                                         acceleration.acceleration);

              specfem::compute::atomic_add_on_device(index, acceleration,
                                                     field);
            });
      });

  Kokkos::fence();
  return;
}

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_TPP
