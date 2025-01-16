#pragma once

#include "algorithms/gradient.hpp"
#include "chunk_element/field.hpp"
#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/medium.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

namespace impl {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, int NGLL>
class helper {
public:
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static auto ngll = NGLL;

  helper(specfem::compute::assembly assembly,
         Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>
             wavefield_on_entire_grid)
      : assembly(assembly), wavefield_on_entire_grid(wavefield_on_entire_grid) {
    if (assembly.mesh.ngllz != ngll || assembly.mesh.ngllx != ngll) {
      throw std::runtime_error("Number of quadrature points not supported");
    }
  }

  void operator()(const specfem::wavefield::type wavefield_type) {
    const auto buffer = assembly.fields.buffer;

    const int nspec = assembly.mesh.nspec;
    const int ngllz = assembly.mesh.ngllz;
    const int ngllx = assembly.mesh.ngllx;

    const auto elements =
        assembly.element_types.get_elements_on_device(medium_tag, property_tag);

    const int nelements = elements.extent(0);

    if (nelements == 0) {
      return;
    }

    using ParallelConfig = specfem::parallel_config::default_chunk_config<
        specfem::dimension::type::dim2,
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;

    using ChunkElementFieldType = specfem::chunk_element::field<
        ParallelConfig::chunk_size, ngll, specfem::dimension::type::dim2,
        medium_tag, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true, true, false,
        false>;

    using QuadratureType = specfem::element::quadrature<
        ngll, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

    using PointPropertyType =
        specfem::point::properties<specfem::dimension::type::dim2, medium_tag,
                                   property_tag, false>;

    using PointFieldDerivativesType =
        specfem::point::field_derivatives<specfem::dimension::type::dim2,
                                          medium_tag, false>;

    using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

    int scratch_size =
        ChunkElementFieldType::shmem_size() + QuadratureType::shmem_size();

    ChunkPolicyType chunk_policy(elements, ngllz, ngllx);

    Kokkos::parallel_for(
        "specfem::compute::assembly::compute_wavefield",
        chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
          QuadratureType quadrature(team);
          ChunkElementFieldType field(team);

          specfem::compute::load_on_device(team, assembly.mesh.quadratures,
                                           quadrature);

          for (int tile = 0; tile < ChunkPolicyType::tile_size;
               tile += ChunkPolicyType::chunk_size) {
            const int starting_element_index =
                team.league_rank() * ChunkPolicyType::tile_size + tile;

            if (starting_element_index >= nelements) {
              break;
            }

            const auto iterator =
                chunk_policy.league_iterator(starting_element_index);
            specfem::compute::load_on_device(team, iterator, buffer, field);
            team.team_barrier();

            const auto sv_wavefield =
                Kokkos::subview(wavefield_on_entire_grid, iterator.get_range(),
                                Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

            specfem::medium::compute_wavefield<MediumTag, PropertyTag>(
                team, iterator, assembly, quadrature, field, wavefield_type,
                sv_wavefield);
          }
        });

    return;
  }

private:
  const specfem::compute::assembly assembly;
  Kokkos::View<type_real ****, Kokkos::LayoutLeft,
               Kokkos::DefaultExecutionSpace>
      wavefield_on_entire_grid;
};

} // namespace impl

// namespace impl {
// template <specfem::element::medium_tag MediumTag,
//           specfem::wavefield::type Component>
// class field_type_parameters;

// template <specfem::element::medium_tag MediumTag>
// class field_type_parameters<MediumTag,
//                             specfem::wavefield::type::displacement> {
// public:
//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto store_displacement = true;
//   constexpr static auto store_velocity = false;
//   constexpr static auto store_acceleration = false;
//   constexpr static auto store_mass_matrix = false;
//   constexpr static auto num_components = 2;
// };

// template <specfem::element::medium_tag MediumTag>
// class field_type_parameters<MediumTag,
//                             specfem::wavefield::type::velocity> {
// public:
//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto store_displacement = false;
//   constexpr static auto store_velocity = true;
//   constexpr static auto store_acceleration = false;
//   constexpr static auto store_mass_matrix = false;
//   constexpr static auto num_components = 2;
// };

// template <specfem::element::medium_tag MediumTag>
// class field_type_parameters<MediumTag,
//                             specfem::wavefield::type::acceleration> {
// public:
//   constexpr static auto medium_tag = MediumTag;
//   constexpr static auto store_displacement = false;
//   constexpr static auto store_velocity = false;
//   constexpr static auto store_acceleration = true;
//   constexpr static auto store_mass_matrix = false;
//   constexpr static auto num_components = 2;
// };

// template <specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag,
//           specfem::wavefield::type Component, int NGLL>
// class helper;

// template <specfem::element::property_tag PropertyTag,
//           specfem::wavefield::type Component, int NGLL>
// class helper<specfem::element::medium_tag::elastic, PropertyTag, Component,
//              NGLL> {
// public:
//   constexpr static auto medium_tag = specfem::element::medium_tag::elastic;
//   constexpr static auto property_tag = PropertyTag;
//   constexpr static auto component = Component;
//   constexpr static auto ngll = NGLL;
//   using field_parameters = field_type_parameters<medium_tag, component>;
//   helper(specfem::compute::assembly assembly,
//          Kokkos::View<type_real ****, Kokkos::LayoutLeft,
//                       Kokkos::DefaultExecutionSpace>
//              wavefield_on_entire_grid)
//       : assembly(assembly),
//       wavefield_on_entire_grid(wavefield_on_entire_grid) {
//     if (assembly.mesh.ngllz != ngll || assembly.mesh.ngllx != ngll) {
//       throw std::runtime_error("Number of quadrature points not supported");
//     }
//     return;
//   }

//   void operator()() const {
//     const auto buffer = assembly.fields.buffer;

//     const int nspec = assembly.mesh.nspec;
//     const int ngllz = assembly.mesh.ngllz;
//     const int ngllx = assembly.mesh.ngllx;

//     const auto elements =
//         assembly.properties.get_elements_on_device(medium_tag, property_tag);

//     const int nelements = elements.extent(0);

//     if (nelements == 0) {
//       return;
//     }

//     using PointFieldType =
//         specfem::point::field<specfem::dimension::type::dim2, medium_tag,
//                               field_parameters::store_displacement,
//                               field_parameters::store_velocity,
//                               field_parameters::store_acceleration,
//                               field_parameters::store_mass_matrix, false>;

//     using simd = specfem::datatype::simd<type_real, false>;
//     constexpr int simd_size = simd::size();

//     using ParallelConfig = specfem::parallel_config::default_chunk_config<
//         specfem::dimension::type::dim2, simd, Kokkos::DefaultExecutionSpace>;

//     using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

//     ChunkPolicyType chunk_policy(elements, ngllz, ngllx);

//     Kokkos::parallel_for(
//         "specfem::domain::impl::kernels::elements::compute_mass_matrix",
//         static_cast<const typename ChunkPolicyType::policy_type &>(
//             chunk_policy),
//         KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type
//         &team) {
//           for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
//                tile += ChunkPolicyType::chunk_size * simd_size) {
//             const int starting_element_index =
//                 team.league_rank() * ChunkPolicyType::tile_size * simd_size +
//                 tile;

//             if (starting_element_index >= nelements) {
//               break;
//             }

//             const auto iterator =
//                 chunk_policy.league_iterator(starting_element_index);

//             Kokkos::parallel_for(
//                 Kokkos::TeamThreadRange(team, iterator.chunk_size()),
//                 [&](const int i) {
//                   const auto iterator_index = iterator(i);
//                   const auto index = iterator_index.index;

//                   PointFieldType field;

//                   specfem::compute::load_on_device(index, buffer, field);

//                   for (int icomponent = 0;
//                        icomponent < field_parameters::num_components;
//                        icomponent++) {
//                     wavefield_on_entire_grid(index.ispec, index.iz, index.ix,
//                                              icomponent) = field(icomponent);
//                   }
//                 });
//           }
//         });

//     return;
//   }

// private:
//   const specfem::compute::assembly assembly;
//   Kokkos::View<type_real ****, Kokkos::LayoutLeft,
//                Kokkos::DefaultExecutionSpace>
//       wavefield_on_entire_grid;
// };

// template <specfem::element::property_tag PropertyTag,
//           specfem::wavefield::type Component, int NGLL>
// class helper<specfem::element::medium_tag::acoustic, PropertyTag, Component,
//              NGLL> {
// public:
//   constexpr static auto medium_tag = specfem::element::medium_tag::acoustic;
//   constexpr static auto property_tag = PropertyTag;
//   constexpr static auto component = Component;
//   constexpr static auto ngll = NGLL;
//   using field_parameters =
//       field_type_parameters<specfem::element::medium_tag::acoustic,
//       Component>;
//   helper(specfem::compute::assembly assembly,
//          Kokkos::View<type_real ****, Kokkos::LayoutLeft,
//                       Kokkos::DefaultExecutionSpace>
//              wavefield_on_entire_grid)
//       : assembly(assembly),
//       wavefield_on_entire_grid(wavefield_on_entire_grid) {
//     if (assembly.mesh.ngllz != ngll || assembly.mesh.ngllx != ngll) {
//       throw std::runtime_error("Number of quadrature points not supported");
//     }
//   }

//   void operator()() const {
//     const auto buffer = assembly.fields.buffer;

//     const int nspec = assembly.mesh.nspec;
//     const int ngllz = assembly.mesh.ngllz;
//     const int ngllx = assembly.mesh.ngllx;

//     const auto elements =
//         assembly.properties.get_elements_on_device(medium_tag, property_tag);
//     const int nelements = elements.extent(0);

//     if (nelements == 0) {
//       return;
//     }

//     constexpr auto num_components = field_parameters::num_components;

//     using simd = specfem::datatype::simd<type_real, false>;
//     constexpr int simd_size = simd::size();
//     using ParallelConfig = specfem::parallel_config::default_chunk_config<
//         specfem::dimension::type::dim2, simd, Kokkos::DefaultExecutionSpace>;

//     using ChunkElementFieldType = specfem::chunk_element::field<
//         ParallelConfig::chunk_size, ngll, specfem::dimension::type::dim2,
//         medium_tag, specfem::kokkos::DevScratchSpace,
//         Kokkos::MemoryTraits<Kokkos::Unmanaged>,
//         field_parameters::store_displacement,
//         field_parameters::store_velocity,
//         field_parameters::store_acceleration,
//         field_parameters::store_mass_matrix, false>;

//     using QuadratureType = specfem::element::quadrature<
//         ngll, specfem::dimension::type::dim2,
//         specfem::kokkos::DevScratchSpace,
//         Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

//     using PointPropertyType =
//         specfem::point::properties<specfem::dimension::type::dim2,
//         medium_tag,
//                                    property_tag, false>;

//     using PointFieldDerivativesType =
//         specfem::point::field_derivatives<specfem::dimension::type::dim2,
//                                           medium_tag, false>;

//     using ChunkPolicyType = specfem::policy::element_chunk<ParallelConfig>;

//     int scratch_size =
//         ChunkElementFieldType::shmem_size() + QuadratureType::shmem_size();
//     ChunkPolicyType chunk_policy(elements, ngllz, ngllx);

//     Kokkos::parallel_for(
//         "compute_wavefield",
//         chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
//         KOKKOS_CLASS_LAMBDA(const typename ChunkPolicyType::member_type
//         &team) {
//           QuadratureType quadrature(team);
//           ChunkElementFieldType field(team);

//           specfem::compute::load_on_device(team, assembly.mesh.quadratures,
//                                            quadrature);

//           for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
//                tile += ChunkPolicyType::chunk_size * simd_size) {
//             const int starting_element_index =
//                 team.league_rank() * ChunkPolicyType::tile_size * simd_size +
//                 tile;

//             if (starting_element_index >= nelements) {
//               break;
//             }

//             const auto iterator =
//                 chunk_policy.league_iterator(starting_element_index);
//             specfem::compute::load_on_device(team, iterator, buffer, field);
//             team.team_barrier();

//             const auto &active_field = [&]() {
//               if constexpr (Component ==
//                             specfem::wavefield::type::displacement) {
//                 return field.displacement;
//               } else if constexpr (Component ==
//                                    specfem::wavefield::type::velocity) {
//                 return field.velocity;
//               } else if constexpr (Component ==
//               specfem::wavefield::type::
//                                                     acceleration) {
//                 return field.acceleration;
//               } else {
//                 static_assert("component not supported");
//               }
//             }();

//             specfem::algorithms::gradient(
//                 team, iterator, assembly.partial_derivatives,
//                 quadrature.hprime_gll, active_field,
//                 [&](const typename ChunkPolicyType::iterator_type::index_type
//                         &iterator_index,
//                     const typename PointFieldDerivativesType::ViewType &du) {
//                   PointPropertyType point_property;

//                   specfem::compute::load_on_device(iterator_index.index,
//                                                    assembly.properties,
//                                                    point_property);

//                   PointFieldDerivativesType field_derivatives(du);

//                   auto stress = specfem::medium::compute_stress(
//                       point_property, field_derivatives);

//                   for (int icomponent = 0; icomponent < num_components;
//                        icomponent++) {
//                     wavefield_on_entire_grid(
//                         iterator_index.index.ispec, iterator_index.index.iz,
//                         iterator_index.index.ix, icomponent) =
//                         stress.T(icomponent, 0);
//                   }
//                 });
//           }
//         });

//     return;
//   }

// private:
//   const specfem::compute::assembly assembly;
//   Kokkos::View<type_real ****, Kokkos::LayoutLeft,
//                Kokkos::DefaultExecutionSpace>
//       wavefield_on_entire_grid;
// };
// } // namespace impl
