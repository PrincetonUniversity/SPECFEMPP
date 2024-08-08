#ifndef _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
#define _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP

// #include "acoustic/acoustic2d.hpp"
#include "algorithms/divergence.hpp"
#include "algorithms/gradient.hpp"
#include "chunk_element/field.hpp"
#include "chunk_element/stress_integrand.hpp"
#include "compute/interface.hpp"
// #include "elastic/elastic2d.hpp"
#include "acoustic/acoustic2d.tpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "elastic/elastic2d.tpp"
#include "element.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    element_kernel_base(
        const specfem::compute::assembly &assembly,
        const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
        const quadrature_points_type &quadrature_points)
    : nelements(h_element_kernel_index_mapping.extent(0)),
      element_kernel_index_mapping("specfem::domain::impl::kernels::element_"
                                   "kernel_base::element_kernel_index_mapping",
                                   nelements),
      h_element_kernel_index_mapping(h_element_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      properties(assembly.properties), boundaries(assembly.boundaries),
      quadrature_points(quadrature_points),
      boundary_values(assembly.boundary_values.get_container<BoundaryTag>()) {

  // Check if the elements being allocated to this kernel are of the correct
  // type
  for (int ispec = 0; ispec < nelements; ispec++) {
    const int ielement = h_element_kernel_index_mapping(ispec);
    if ((assembly.properties.h_element_types(ielement) != MediumTag) &&
        (assembly.properties.h_element_property(ielement) != PropertyTag)) {
      throw std::runtime_error("Invalid element detected in kernel");
    }
  }

  // Assert that ispec of the elements is contiguous
  for (int ispec = 0; ispec < nelements; ispec++) {
    if (ispec != 0) {
      if (h_element_kernel_index_mapping(ispec) !=
          h_element_kernel_index_mapping(ispec - 1) + 1) {
        throw std::runtime_error("Element index mapping is not contiguous");
      }
    }
  }

  Kokkos::deep_copy(element_kernel_index_mapping,
                    h_element_kernel_index_mapping);
  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    compute_mass_matrix(
        const type_real dt,
        const specfem::compute::simulation_field<WavefieldType> &field) const {

  constexpr bool using_simd = true;
  constexpr int components = medium_type::components;
  using PointMassType = specfem::point::field<DimensionType, MediumTag, false,
                                              false, false, true, using_simd>;

  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;

  using simd = specfem::datatype::simd<type_real, using_simd>;

  constexpr int simd_size = simd::size();
  using ParallelConfig = specfem::parallel_config::default_chunk_config<simd>;

  using ChunkPolicyType =
      specfem::policy::element_chunk<ParallelConfig,
                                     Kokkos::DefaultExecutionSpace>;

  using PointBoundaryType = specfem::point::boundary<BoundaryTag, using_simd>;

  constexpr int NGLL = quadrature_points_type::NGLL;

  ChunkPolicyType chunk_policy(element_kernel_index_mapping, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_mass_matrix",
      chunk_policy.get_policy(),
      KOKKOS_CLASS_LAMBDA(const ChunkPolicyType::member_type &team) {
        for (int tile = 0; tile < ChunkPolicyType::TileSize * simd_size;
             tile += ChunkPolicyType::ChunkSize * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::TileSize * simd_size + tile;

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
                const int ix = iterator_index.index.ix;
                const int iz = iterator_index.index.iz;

                // specfem::point::properties<DimensionType, MediumTag,
                // PropertyTag>
                //     point_property;

                const auto point_property = [&]()
                    -> specfem::point::properties<DimensionType, MediumTag,
                                                  PropertyTag, using_simd> {
                  specfem::point::properties<DimensionType, MediumTag,
                                             PropertyTag, using_simd>
                      point_property;

                  specfem::compute::load_on_device(index, properties,
                                                   point_property);
                  return point_property;
                }();

                // specfem::point::partial_derivatives2<true>
                // point_partial_derivatives;

                const auto point_partial_derivatives = [&]()
                    -> specfem::point::partial_derivatives2<using_simd, true> {
                  specfem::point::partial_derivatives2<using_simd, true>
                      point_partial_derivatives;
                  specfem::compute::load_on_device(index, partial_derivatives,
                                                   point_partial_derivatives);
                  return point_partial_derivatives;
                }();

                PointMassType mass_matrix =
                    specfem::domain::impl::elements::mass_matrix_component(
                        point_property, point_partial_derivatives);

                for (int icomp = 0; icomp < components; icomp++) {
                  mass_matrix.mass_matrix(icomp) *= wgll(ix) * wgll(iz);
                }

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                specfem::domain::impl::boundary_conditions::
                    compute_mass_matrix_terms(dt, point_boundary,
                                              point_property, mass_matrix);

                specfem::compute::atomic_add_on_device(index, mass_matrix,
                                                       field);
              });
        }
      });

  Kokkos::fence();

  return;
}

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag,
//           specfem::element::boundary_tag BoundaryTag,
//           typename quadrature_points_type>
// void specfem::domain::impl::kernels::element_kernel_base<
//     WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
//     quadrature_points_type>::
//     compute_mass_matrix(
//         const specfem::compute::simulation_field<WavefieldType> &field) const
//         {
//   constexpr int components = medium_type::components;
//   using PointMassType = specfem::point::field<DimensionType, MediumTag,
//   false,
//                                               false, false, true>;

//   if (nelements == 0)
//     return;

//   const auto wgll = quadrature.gll.weights;

//   Kokkos::parallel_for(
//       "specfem::domain::kernes::elements::compute_mass_matrix",
//       specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         int ngllx, ngllz;
//         quadrature_points.get_ngll(&ngllx, &ngllz);
//         const auto ispec_l =
//             element_kernel_index_mapping(team_member.league_rank());

//         Kokkos::parallel_for(
//             quadrature_points.template
//             TeamThreadRange<specfem::enums::axes::x,
//                                                        specfem::enums::axes::z>(
//                 team_member),
//             [&](const int xz) {
//               int ix, iz;
//               sub2ind(xz, ngllx, iz, ix);

//               const specfem::point::index index(ispec_l, iz, ix);

//               const auto point_property =
//                   [&]() -> specfem::point::properties<MediumTag, PropertyTag>
//                   {
//                 specfem::point::properties<MediumTag, PropertyTag>
//                     point_property;

//                 specfem::compute::load_on_device(index, properties,
//                                                  point_property);
//                 return point_property;
//               }();

//               const auto point_partial_derivatives =
//                   [&]() -> specfem::point::partial_derivatives2<true> {
//                 specfem::point::partial_derivatives2<true>
//                     point_partial_derivatives;
//                 specfem::compute::load_on_device(index, partial_derivatives,
//                                                  point_partial_derivatives);
//                 return point_partial_derivatives;
//               }();

//               PointMassType point_mass;

//               element.compute_mass_matrix_component(point_property,
//                                                     point_partial_derivatives,
//                                                     point_mass.mass_matrix);

//               for (int icomponent = 0; icomponent < components; icomponent++)
//               {
//                 point_mass.mass_matrix[icomponent] *= wgll(ix) * wgll(iz);
//               }

//               specfem::compute::atomic_add_on_device(index, point_mass,
//               field);
//             });
//       });

//   Kokkos::fence();

//   return;
// }

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag,
//           specfem::element::boundary_tag BoundaryTag,
//           typename quadrature_points_type>
// template <specfem::enums::time_scheme::type time_scheme>
// void specfem::domain::impl::kernels::element_kernel_base<
//     WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
//     quadrature_points_type>::
//     mass_time_contribution(
//         const type_real dt,
//         const specfem::compute::simulation_field<WavefieldType> &field) const
//         {

//   constexpr int components = medium_type::components;
//   using PointMassType = specfem::point::field<DimensionType, MediumTag,
//   false,
//                                               false, false, true>;

//   if (nelements == 0)
//     return;

//   const auto wgll = quadrature.gll.weights;
//   const auto index_mapping = points.index_mapping;

//   Kokkos::parallel_for(
//       "specfem::domain::kernes::elements::add_mass_matrix_contribution",
//       specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
//       KOKKOS_CLASS_LAMBDA(
//           const specfem::kokkos::DeviceTeam::member_type &team_member) {
//         int ngllx, ngllz;
//         quadrature_points.get_ngll(&ngllx, &ngllz);
//         const auto ispec_l =
//             element_kernel_index_mapping(team_member.league_rank());

//         const auto point_boundary_type = boundary_conditions(ispec_l);

//         Kokkos::parallel_for(
//             quadrature_points.template
//             TeamThreadRange<specfem::enums::axes::x,
//                                                        specfem::enums::axes::z>(
//                 team_member),
//             [&](const int xz) {
//               int ix, iz;
//               sub2ind(xz, ngllx, iz, ix);

//               const specfem::point::index index(ispec_l, iz, ix);

//               const auto point_property =
//                   [&]() -> specfem::point::properties<MediumTag, PropertyTag>
//                   {
//                 specfem::point::properties<MediumTag, PropertyTag>
//                     point_property;

//                 specfem::compute::load_on_device(index, properties,
//                                                  point_property);
//                 return point_property;
//               }();

//               const auto point_partial_derivatives =
//                   [&]() -> specfem::point::partial_derivatives2<true> {
//                 specfem::point::partial_derivatives2<true>
//                     point_partial_derivatives;
//                 specfem::compute::load_on_device(index, partial_derivatives,
//                                                  point_partial_derivatives);
//                 return point_partial_derivatives;
//               }();

//               PointMassType point_mass;

//               specfem::kokkos::array_type<type_real, dimension::dim> weight(
//                   wgll(ix), wgll(iz));

//               element.template mass_time_contribution<time_scheme>(
//                   xz, dt, weight, point_partial_derivatives, point_property,
//                   point_boundary_type, point_mass.mass_matrix);

//               specfem::compute::atomic_add_on_device(index, point_mass,
//               field);
//             });
//       });

//   Kokkos::fence();
//   return;
// }

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    compute_stiffness_interaction(
        const int istep,
        const specfem::compute::simulation_field<WavefieldType> &field) const {

  constexpr int NumberOfDimensions =
      specfem::dimension::dimension<DimensionType>::dim;
  constexpr int components = medium_type::components;
  // Number of quadrature points
  constexpr int NGLL = quadrature_points_type::NGLL;
  constexpr bool using_simd = true;

  using simd = specfem::datatype::simd<type_real, using_simd>;

  constexpr int simd_size = simd::size();

  using ParallelConfig = specfem::parallel_config::default_chunk_config<simd>;
  // Element field type - represents which fields to fetch from global field
  // struct
  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, NGLL, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;

  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      ParallelConfig::chunk_size, NGLL, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      using_simd>;

  // Quadrature type - represents data structure used to store element
  // quadrature
  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;
  // Data structure used to field at GLL point - represents which field to
  // atomically update
  using PointAccelerationType =
      specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                            using_simd>;
  using PointVelocityType =
      specfem::point::field<DimensionType, MediumTag, false, true, false, false,
                            using_simd>;

  using PointBoundaryType = specfem::point::boundary<BoundaryTag, using_simd>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionType, MediumTag, using_simd>;

  if (nelements == 0)
    return;

  const auto hprime = quadrature.gll.hprime;
  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  using ChunkPolicyType =
      specfem::policy::element_chunk<ParallelConfig,
                                     Kokkos::DefaultExecutionSpace>;

  ChunkPolicyType chunk_policy(element_kernel_index_mapping, NGLL, NGLL);

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_interaction",
      chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(const ChunkPolicyType::member_type &team) {
        ChunkElementFieldType element_field(team);
        ElementQuadratureType element_quadrature(team);
        ChunkStressIntegrandType stress_integrand(team);

        specfem::compute::load_on_device(team, quadrature, element_quadrature);
        for (int tile = 0; tile < ChunkPolicyType::TileSize * simd_size;
             tile += ChunkPolicyType::ChunkSize * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::TileSize * simd_size + tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);
          specfem::compute::load_on_device(team, iterator, field,
                                           element_field);

          team.team_barrier();

          specfem::algorithms::gradient(
              team, iterator, partial_derivatives,
              element_quadrature.hprime_gll, element_field.displacement,
              // Compute stresses using the gradients
              [&](const typename ChunkPolicyType::iterator_type::index_type
                      &iterator_index,
                  const typename PointFieldDerivativesType::ViewType &du) {
                const auto index = iterator_index.index;

                specfem::point::partial_derivatives2<using_simd, false>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);

                specfem::point::properties<DimensionType, MediumTag,
                                           PropertyTag, using_simd>
                    point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);
                // const auto point_partial_derivatives = [&]()
                //     -> specfem::point::partial_derivatives2<using_simd,
                //     false> {
                //   specfem::point::partial_derivatives2<using_simd, false>
                //       point_partial_derivatives;
                //   specfem::compute::load_on_device(index,
                //   partial_derivatives,
                //                                    point_partial_derivatives);
                //   return point_partial_derivatives;
                // }();

                // const auto point_property = [&]()
                //     -> specfem::point::properties<DimensionType, MediumTag,
                //                                   PropertyTag, using_simd> {
                //   specfem::point::properties<DimensionType, MediumTag,
                //                              PropertyTag, using_simd>
                //       point_property;

                //   specfem::compute::load_on_device(index, properties,
                //                                    point_property);
                //   return point_property;
                // }();

                PointFieldDerivativesType field_derivatives(du);

                const auto point_stress_integrand =
                    specfem::domain::impl::elements::compute_stress_integrands(
                        point_partial_derivatives, point_property,
                        field_derivatives);

                const int ielement = iterator_index.ielement;

                for (int idim = 0; idim < NumberOfDimensions; ++idim) {
                  for (int icomponent = 0; icomponent < components;
                       ++icomponent) {
                    stress_integrand.F(ielement, index.iz, index.ix, idim,
                                       icomponent) =
                        point_stress_integrand.F(idim, icomponent);
                  }
                }

                // typename specfem::datatype::simd<type_real,
                // using_simd>::datatype
                //     dummy = 0.0;

                // for (int icomponent = 0; icomponent < components;
                //      ++icomponent) {
                //   for (int idim = 0; idim < NumberOfDimensions; ++idim) {
                //     dummy += point_stress_integrand.F(idim, icomponent);
                //   }
                // }

                // stress_integrand.F(0, 0, 0, 0, 0) = dummy;
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              team, iterator, partial_derivatives, wgll,
              element_quadrature.hprime_wgll, stress_integrand.F,
              [&](const typename ChunkPolicyType::iterator_type::index_type
                      &iterator_index,
                  const typename PointAccelerationType::ViewType &result) {
                auto index = iterator_index.index;
                PointAccelerationType acceleration(result);

                for (int icomponent = 0; icomponent < components;
                     icomponent++) {
                  acceleration.acceleration(icomponent) *=
                      static_cast<type_real>(-1.0);
                }

                specfem::point::properties<DimensionType, MediumTag,
                                           PropertyTag, using_simd>
                    point_property;
                specfem::compute::load_on_device(index, properties,
                                                 point_property);

                PointVelocityType velocity;
                specfem::compute::load_on_device(index, field, velocity);

                PointBoundaryType point_boundary;
                specfem::compute::load_on_device(index, boundaries,
                                                 point_boundary);

                specfem::domain::impl::boundary_conditions::
                    apply_boundary_conditions(point_boundary, point_property,
                                              velocity, acceleration);

                specfem::compute::atomic_add_on_device(index, acceleration,
                                                       field);
              });
        }
      });

  Kokkos::fence();

  //   Kokkos::parallel_for(
  //       "specfem::domain::impl::kernels::elements::compute_stiffness_"
  //       "interaction",
  //       specfem::kokkos::DeviceTeam(nelements, NTHREADS, NLANES)
  //           .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
  //       KOKKOS_CLASS_LAMBDA(
  //           const specfem::kokkos::DeviceTeam::member_type &team_member) {
  //         int ngllx, ngllz;
  //         quadrature_points.get_ngll(&ngllx, &ngllz);
  //         const auto ispec_l =
  //             element_kernel_index_mapping(team_member.league_rank());

  //         const auto point_boundary_type = boundary_conditions(ispec_l);

  //         // Instantiate shared views
  //         // ---------------------------------------------------------------
  //         ElementFieldType element_field(team_member);
  //         ElementQuadratureType element_quadrature(team_member);
  //         ElementFieldViewType
  //         s_stress_integrand_xi(team_member.team_scratch(0));
  //         ElementFieldViewType s_stress_integrand_gamma(
  //             team_member.team_scratch(0));

  //         // ---------- Allocate shared views -------------------------------
  //         specfem::compute::load_on_device(team_member, quadrature,
  //                                          element_quadrature);
  //         specfem::compute::load_on_device(team_member, ispec_l, field,
  //                                          element_field);
  //         // ---------------------------------------------------------------

  //         team_member.team_barrier();

  //         Kokkos::parallel_for(
  //             quadrature_points.template
  //             TeamThreadRange<specfem::enums::axes::x,
  //                                                        specfem::enums::axes::z>(
  //                 team_member),
  //             [&](const int xz) {
  //               int ix, iz;
  //               sub2ind(xz, ngllx, iz, ix);

  //               specfem::kokkos::array_type<type_real,
  //               medium_type::components>
  //                   dudxl;
  //               specfem::kokkos::array_type<type_real,
  //               medium_type::components>
  //                   dudzl;

  //               const specfem::point::index index(ispec_l, iz, ix);

  //               const auto point_partial_derivatives =
  //                   [&]() -> specfem::point::partial_derivatives2<true> {
  //                 specfem::point::partial_derivatives2<true>
  //                     point_partial_derivatives;
  //                 specfem::compute::load_on_device(index,
  //                 partial_derivatives,
  //                                                  point_partial_derivatives);
  //                 return point_partial_derivatives;
  //               }();

  //               element.compute_gradient(
  //                   xz, element_quadrature.hprime_gll,
  //                   element_field.displacement, point_partial_derivatives,
  //                   point_boundary_type, dudxl, dudzl);

  //               specfem::kokkos::array_type<type_real,
  //               medium_type::components>
  //                   stress_integrand_xi;
  //               specfem::kokkos::array_type<type_real,
  //               medium_type::components>
  //                   stress_integrand_gamma;

  //               const auto point_property =
  //                   [&]() -> specfem::point::properties<MediumTag,
  //                   PropertyTag> {
  //                 specfem::point::properties<MediumTag, PropertyTag>
  //                     point_property;

  //                 specfem::compute::load_on_device(index, properties,
  //                                                  point_property);
  //                 return point_property;
  //               }();

  //               element.compute_stress(xz, dudxl, dudzl,
  //                                      point_partial_derivatives,
  //                                      point_property, point_boundary_type,
  //                                      stress_integrand_xi,
  //                                      stress_integrand_gamma);
  // #ifdef KOKKOS_ENABLE_CUDA
  // #pragma unroll
  // #endif
  //               for (int icomponent = 0; icomponent < components;
  //               ++icomponent) {
  //                 s_stress_integrand_xi(iz, ix, icomponent) =
  //                     stress_integrand_xi[icomponent];
  //                 s_stress_integrand_gamma(iz, ix, icomponent) =
  //                     stress_integrand_gamma[icomponent];
  //               }
  //             });

  //         team_member.team_barrier();

  //         Kokkos::parallel_for(
  //             quadrature_points.template
  //             TeamThreadRange<specfem::enums::axes::x,
  //                                                        specfem::enums::axes::z>(
  //                 team_member),
  //             [&, istep](const int xz) {
  //               int iz, ix;
  //               sub2ind(xz, ngllx, iz, ix);
  //               constexpr auto tag = boundary_conditions_type::value;

  //               const specfem::kokkos::array_type<type_real, dimension::dim>
  //                   weight(wgll(ix), wgll(iz));

  //               PointAccelerationType acceleration;

  //               // Get velocity, partial derivatives, and properties
  //               // only if needed by the boundary condition
  //               //
  //               ---------------------------------------------------------------
  //               constexpr bool load_boundary_variables =
  //                   ((tag == specfem::element::boundary_tag::stacey) ||
  //                    (tag == specfem::element::boundary_tag::
  //                                composite_stacey_dirichlet));

  //               constexpr bool store_boundary_values =
  //                   ((BoundaryTag == specfem::element::boundary_tag::stacey)
  //                   &&
  //                    (WavefieldType == specfem::wavefield::type::forward));

  //               const specfem::point::index index(ispec_l, iz, ix);

  //               const auto velocity = [&]() -> PointVelocityType {
  //                 if constexpr (load_boundary_variables) {
  //                   PointVelocityType velocity_l;
  //                   specfem::compute::load_on_device(index, field,
  //                   velocity_l); return velocity_l;
  //                 } else {
  //                   return {};
  //                 }
  //               }();

  //               const auto point_partial_derivatives =
  //                   [&]() -> specfem::point::partial_derivatives2<true> {
  //                 if constexpr (load_boundary_variables) {
  //                   specfem::point::partial_derivatives2<true>
  //                       point_partial_derivatives;
  //                   specfem::compute::load_on_device(index,
  //                   partial_derivatives,
  //                                                    point_partial_derivatives);
  //                   return point_partial_derivatives;

  //                 } else {
  //                   return {};
  //                 }
  //               }();

  //               const auto point_property =
  //                   [&]() -> specfem::point::properties<MediumTag,
  //                   PropertyTag> {
  //                 if constexpr (load_boundary_variables) {
  //                   specfem::point::properties<MediumTag, PropertyTag>
  //                       point_property;
  //                   specfem::compute::load_on_device(index, properties,
  //                                                    point_property);
  //                   return point_property;
  //                 } else {
  //                   return specfem::point::properties<MediumTag,
  //                   PropertyTag>();
  //                 }
  //               }();
  //               //
  //               ---------------------------------------------------------------

  //               element.compute_acceleration(
  //                   xz, weight, s_stress_integrand_xi,
  //                   s_stress_integrand_gamma, element_quadrature.hprimew_gll,
  //                   point_partial_derivatives, point_property,
  //                   point_boundary_type, velocity.velocity,
  //                   acceleration.acceleration);

  //               if constexpr (store_boundary_values) {
  //                 specfem::compute::store_on_device(istep, index,
  //                 acceleration,
  //                                                   boundary_values);
  //               }

  //               specfem::compute::atomic_add_on_device(index, acceleration,
  //                                                      field);

  //               // #ifdef KOKKOS_ENABLE_CUDA
  //               // #pragma unroll
  //               // #endif
  //               //               for (int icomponent = 0; icomponent <
  //               components;
  //               //               ++icomponent) {
  //               // Kokkos::single(Kokkos::PerThread(team_member),
  //               //                 [&]() {
  //               // Kokkos::atomic_add(&field.field_dot_dot(iglob,
  //               //                   icomponent),
  //               // acceleration[icomponent]);
  //               //                 });
  //               //               }
  //             });
  //       });

  //   Kokkos::fence();

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType,
          specfem::element::property_tag PropertyTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel<
    specfem::wavefield::type::backward, DimensionType, MediumType, PropertyTag,
    specfem::element::boundary_tag::stacey,
    quadrature_points_type>::compute_stiffness_interaction(const int istep)
    const {

  constexpr int components =
      specfem::medium::medium<DimensionType, MediumType>::components;
  constexpr bool using_simd = false;
  // Number of quadrature points
  using PointAccelerationType =
      specfem::point::field<DimensionType, MediumType, false, false, true,
                            false, using_simd>;

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_"
      "interaction",
      specfem::kokkos::DeviceTeam(this->nelements, NTHREADS, NLANES),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        this->quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            this->element_kernel_index_mapping(team_member.league_rank());

        Kokkos::parallel_for(
            this->quadrature_points.template TeamThreadRange<
                specfem::enums::axes::z, specfem::enums::axes::x>(team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              const specfem::point::index index(ispec_l, iz, ix);

              PointAccelerationType acceleration;
              specfem::compute::load_on_device(
                  istep, index, this->boundary_values, acceleration);

              specfem::compute::atomic_add_on_device(index, acceleration,
                                                     field);
            });
      });
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
