#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"

#include "impl_chunk.hpp"
#include "impl_load.hpp"
#include "impl_policy.hpp"
#include "impl_stress.hpp"

#include "compute/assembly/assembly.hpp"
#include "datatypes/simd.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace benchmarks {

constexpr static auto dimension = specfem::dimension::type::dim2;
constexpr static auto wavefield = specfem::wavefield::simulation_field::forward;
constexpr static auto ngll = 5;
constexpr static auto boundary_tag = specfem::element::boundary_tag::none;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_stiffness_interaction(const specfem::compute::assembly &assembly,
                                   const int &istep, const int &flag) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, boundary_tag);

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;
  const auto &properties = assembly.properties;
  const auto field = assembly.fields.get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;
  const auto boundary_values =
      assembly.boundary_values.get_container<boundary_tag>();

  constexpr bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  constexpr int chunk_size = parallel_config::chunk_size;

  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components();
  constexpr int num_dimensions =
      specfem::element::attributes<dimension, medium_tag>::dimension();

  using ChunkPolicyType = element_chunk<parallel_config>;
  using ChunkElementFieldType = specfem::benchmarks::chunk_field<
      parallel_config::chunk_size, ngll, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;
  using ChunkStressIntegrandType = specfem::benchmarks::chunk_stress_integrand<
      parallel_config::chunk_size, ngll, dimension, medium_tag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      using_simd>;
  using ElementQuadratureType = specfem::element::quadrature<
      ngll, dimension, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;

  using PointAccelerationType =
      specfem::point::field<dimension, medium_tag, false, false, true, false,
                            using_simd>;
  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<dimension, medium_tag, using_simd>;

  using VectorPointViewType =
      specfem::datatype::VectorPointViewType<type_real, 2, components,
                                             using_simd>;
  using MemberType = const typename ChunkPolicyType::member_type;

  using ScalarPointViewType =
      specfem::datatype::ScalarPointViewType<type_real, components, using_simd>;

  constexpr bool is_host_space =
      std::is_same<typename MemberType::execution_space::memory_space,
                   Kokkos::HostSpace>::value;

  using datatype = typename simd::datatype;

  const auto wgll = assembly.mesh.quadratures.gll.weights;

  int scratch_size = ChunkElementFieldType::shmem_size() +
                     ChunkStressIntegrandType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  ChunkPolicyType chunk_policy(elements, ngll, ngll);

  constexpr int simd_size = simd::size();

  Kokkos::parallel_for(
      "specfem::kernels::impl::domain_kernels::compute_stiffness_interaction",
      chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const typename ChunkPolicyType::member_type &team) {
        ChunkElementFieldType element_field(team);
        ElementQuadratureType element_quadrature(team);
        ChunkStressIntegrandType stress_integrand(team);

        specfem::benchmarks::load_quadrature(team, quadrature,
                                             element_quadrature);

        for (int tile = 0; tile < ChunkPolicyType::tile_size * simd_size;
             tile += ChunkPolicyType::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicyType::tile_size * simd_size +
              tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);
          specfem::benchmarks::load_wavefield(team, iterator, field,
                                              element_field);

          team.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int &i) {
                const auto iterator_index = iterator(i);
                const auto &index = iterator_index.index;
                const int &ielement = iterator_index.ielement;
                const int &ix = index.ix;
                const int &iz = index.iz;
                const auto &f = element_field.displacement;
                const auto &quadrature = element_quadrature.hprime_gll;

                datatype df_dxi[components] = { 0.0 };
                datatype df_dgamma[components] = { 0.0 };

                for (int l = 0; l < ngll; ++l) {
                  for (int icomponent = 0; icomponent < components;
                       ++icomponent) {
                    df_dxi[icomponent] +=
                        quadrature(ix, l) * f(ielement, iz, l, icomponent);
                    df_dgamma[icomponent] +=
                        quadrature(iz, l) * f(ielement, l, ix, icomponent);
                  }
                }

                specfem::point::partial_derivatives<dimension, false,
                                                    using_simd>
                    point_partial_derivatives;

                specfem::benchmarks::load_partial_derivative(
                    index, partial_derivatives, point_partial_derivatives);

                VectorPointViewType df;

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  df(0, icomponent) =
                      point_partial_derivatives.xix * df_dxi[icomponent] +
                      point_partial_derivatives.gammax * df_dgamma[icomponent];

                  df(1, icomponent) =
                      point_partial_derivatives.xiz * df_dxi[icomponent] +
                      point_partial_derivatives.gammaz * df_dgamma[icomponent];
                }

                specfem::point::partial_derivatives<dimension, true, using_simd>
                    point_partial_derivatives2;
                specfem::benchmarks::load_partial_derivative(
                    index, partial_derivatives, point_partial_derivatives2);

                PointPropertyType point_property;
                specfem::benchmarks::load_point_property(index, properties,
                                                         point_property);

                PointFieldDerivativesType field_derivatives(df);

                const auto point_stress = specfem::benchmarks::compute_stress(
                    point_property, field_derivatives);

                const auto F = point_stress * point_partial_derivatives2;

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  for (int idim = 0; idim < num_dimensions; ++idim) {
                    stress_integrand.F(ielement, index.iz, index.ix, idim,
                                       icomponent) = F(idim, icomponent);
                  }
                }
              });

          team.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, iterator.chunk_size()),
              [&](const int i) {
                const auto iterator_index = iterator(i);
                const auto &index = iterator_index.index;
                const int ielement = iterator_index.ielement;
                const int ispec = iterator_index.index.ispec;
                const int iz = iterator_index.index.iz;
                const int ix = iterator_index.index.ix;
                const auto &weights = wgll;
                const auto &hprimewgll = element_quadrature.hprime_wgll;
                const auto &f = stress_integrand.F;

                const datatype jacobian =
                    (is_host_space)
                        ? partial_derivatives.h_jacobian(ispec, iz, ix)
                        : partial_derivatives.jacobian(ispec, iz, ix);

                datatype temp1l[components] = { 0.0 };
                datatype temp2l[components] = { 0.0 };

                for (int l = 0; l < ngll; ++l) {
                  for (int icomp = 0; icomp < components; ++icomp) {
                    temp1l[icomp] += f(ielement, iz, l, 0, icomp) *
                                     hprimewgll(ix, l) * jacobian;
                  }
                  for (int icomp = 0; icomp < components; ++icomp) {
                    temp2l[icomp] += f(ielement, l, ix, 1, icomp) *
                                     hprimewgll(iz, l) * jacobian;
                  }
                }

                ScalarPointViewType result;

                for (int icomp = 0; icomp < components; ++icomp) {
                  result(icomp) =
                      weights(iz) * temp1l[icomp] + weights(ix) * temp2l[icomp];
                }

                PointAccelerationType acceleration(result);

                for (int icomponent = 0; icomponent < components;
                     ++icomponent) {
                  acceleration.acceleration(icomponent) *=
                      static_cast<type_real>(-1.0);
                }

                specfem::benchmarks::atomic_add_on_device(index, acceleration,
                                                          field);
              });
        }
      });

  Kokkos::fence();

  return;
}

} // namespace benchmarks
} // namespace specfem
