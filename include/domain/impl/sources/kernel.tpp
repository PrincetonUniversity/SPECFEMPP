#ifndef _DOMAIN_IMPL_SOURCES_KERNEL_TPP
#define _DOMAIN_IMPL_SOURCES_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/sources/acoustic/interface.hpp"
#include "domain/impl/sources/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::source_kernel(const specfem::compute::assembly &assembly)
    : sources(assembly.sources),
      field(assembly.fields.get_simulation_field<WavefieldType>()) {

  this->elements = sources.get_elements_on_device(medium_tag, wavefield_tag);
  return;
}

template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
void specfem::domain::impl::kernels::source_kernel<
    WavefieldType, DimensionType, MediumTag, PropertyTag,
    NGLL>::compute_source_interaction(const int timestep) {

  sources.update_timestep(timestep);

  const int nelements = elements.size();

  if (nelements == 0)
    return;

  using PointSourcesType =
      specfem::point::source<dimension, medium_tag, wavefield_tag>;

  using simd = specfem::datatype::simd<type_real, false>;
  constexpr int simd_size = simd::size();

  using ParallelConfig =
      specfem::parallel_config::default_chunk_config<dimension, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;

  ChunkPolicy chunk_policy(elements, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_mass_matrix",
      static_cast<const typename ChunkPolicy::policy_type &>(chunk_policy),
      KOKKOS_CLASS_LAMBDA(const typename ChunkPolicy::member_type &team) {
        for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
             tile += ChunkPolicy::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicy::tile_size * simd_size +
              tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int i) {
                const auto iterator_index = iterator(i);
                const auto index = iterator_index.index;

                PointSourcesType point_source;
                specfem::compute::load_on_device(index, sources, point_source);

                const auto acceleration = point_source.compute_acceleration();

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        }
      });

  Kokkos::fence();
}

// template <specfem::wavefield::simulation_field WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag, int NGLL>
// void specfem::domain::impl::kernels::source_kernel<
//     WavefieldType, DimensionType, MediumTag, PropertyTag,
//     qp_type>::compute_source_interaction(const int timestep) const {

//   using PointFieldType = specfem::point::field<dimension, medium_tag, false,
//                                                false, true, false,
//                                                using_simd>;

//   if (nsources == 0)
//     return;

//   const auto source_timestep = get_source_timestep(timestep);

//   const auto index_mapping = points.index_mapping;

//   Kokkos::parallel_for(
//       "specfem::domain::domain::compute_source_interaction",
//       specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         int ngllx, ngllz;
//         quadrature_points.get_ngll(&ngllx, &ngllz);
//         const int isource_l =
//             source_domain_index_mapping(team_member.league_rank());
//         const int ispec_l = sources.source_index_mapping(isource_l);

//         Kokkos::parallel_for(
//             quadrature_points.template
//             TeamThreadRange<specfem::enums::axes::z,
//                                                        specfem::enums::axes::x>(
//                 team_member),
//             [=](const int xz) {
//               int iz, ix;
//               sub2ind(xz, ngllx, iz, ix);
//               specfem::point::index<DimensionType> index(ispec_l, iz, ix);

//               const specfem::datatype::ScalarPointViewType<
//                   type_real, components, using_simd>
//                   lagrange_interpolant(Kokkos::subview(
//                       sources.source_array, isource_l, Kokkos::ALL, iz, ix));

//               // Source time function
//               // For acoustic medium, forward simulation, divide by kappa
//               const auto stf = [&, timestep]() {
//                 if constexpr ((WavefieldType ==
//                                specfem::wavefield::simulation_field::forward)
//                                &&
//                               (MediumTag ==
//                                specfem::element::medium_tag::acoustic)) {
//                   const auto point_properties = [&]()
//                       -> specfem::point::properties<DimensionType, MediumTag,
//                                                     PropertyTag, using_simd>
//                                                     {
//                     specfem::point::properties<DimensionType, MediumTag,
//                                                PropertyTag, using_simd>
//                         point_properties;
//                     specfem::compute::load_on_device(index, properties,
//                                                      point_properties);
//                     return point_properties;
//                   }();
//                   specfem::datatype::ScalarPointViewType<type_real,
//                   components,
//                                                          using_simd>
//                       stf(Kokkos::subview(sources.source_time_function,
//                                           timestep, isource_l, Kokkos::ALL));
//                   for (int i = 0; i < components; i++) {
//                     stf(i) = stf(i) / point_properties.kappa;
//                   }
//                   return stf;
//                 } else {
//                   return specfem::datatype::ScalarPointViewType<
//                       type_real, components, using_simd>(
//                       Kokkos::subview(sources.source_time_function, timestep,
//                                       isource_l, Kokkos::ALL));
//                 }
//               }();

//               PointFieldType acceleration;

//               source.compute_interaction(stf, lagrange_interpolant,
//                                          acceleration.acceleration);

//               specfem::compute::atomic_add_on_device(index, acceleration,
//                                                      field);
//             });
//       });

//   Kokkos::fence();
//   return;
// }

#endif // _DOMAIN_IMPL_SOURCES_KERNEL_TPP
