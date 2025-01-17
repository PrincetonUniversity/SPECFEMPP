#pragma once

#include "algorithms/gradient.hpp"
#include "chunk_element/field.hpp"
#include "compute/assembly/assembly.hpp"
#include "compute_material_derivatives.hpp"
#include "medium/compute_frechet_derivatives.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "policies/chunk.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void specfem::kokkos_kernels::impl::compute_material_derivatives(
    const specfem::compute::assembly &assembly, const type_real &dt) {
  auto &properties = assembly.properties;
  auto &kernels = assembly.kernels;
  auto &adjoint_field = assembly.fields.adjoint;
  auto &backward_field = assembly.fields.backward;
  auto &quadrature = assembly.mesh.quadratures;
  auto &partial_derivatives = assembly.partial_derivatives;

  const auto element_index =
      assembly.element_types.get_elements_on_device(MediumTag, PropertyTag);

  const int nelements = element_index.size();

  if (nelements == 0) {
    return;
  }

  constexpr static bool using_simd = true;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, NGLL, DimensionType, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;

  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, DimensionType, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  using AdjointPointFieldType =
      specfem::point::field<DimensionType, MediumTag, false, false, true, false,
                            using_simd>;

  using BackwardPointFieldType =
      specfem::point::field<DimensionType, MediumTag, true, false, false, false,
                            using_simd>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionType, MediumTag, using_simd>;

  using PointPropertiesType =
      specfem::point::properties<DimensionType, MediumTag, PropertyTag,
                                 using_simd>;

  using ChunkPolicy = specfem::policy::element_chunk<ParallelConfig>;

  int scratch_size = 2 * ChunkElementFieldType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  ChunkPolicy chunk_policy(element_index, NGLL, NGLL);

  constexpr int simd_size = simd::size();

  Kokkos::parallel_for(
      "specfem::frechet_derivatives::frechet_elements::compute",
      chunk_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const typename ChunkPolicy::member_type &team) {
        // Allocate scratch memory
        ChunkElementFieldType adjoint_element_field(team);
        ChunkElementFieldType backward_element_field(team);
        ElementQuadratureType quadrature_element(team);

        specfem::compute::load_on_device(team, quadrature, quadrature_element);

        for (int tile = 0; tile < ChunkPolicy::tile_size * simd_size;
             tile += ChunkPolicy::chunk_size * simd_size) {
          const int starting_element_index =
              team.league_rank() * ChunkPolicy::tile_size * simd_size + tile;

          if (starting_element_index >= nelements) {
            break;
          }

          const auto iterator =
              chunk_policy.league_iterator(starting_element_index);

          // Populate Scratch Views
          specfem::compute::load_on_device(team, iterator, adjoint_field,
                                           adjoint_element_field);
          specfem::compute::load_on_device(team, iterator, backward_field,
                                           backward_element_field);

          team.team_barrier();

          // Gernerate the Kernels
          // We call the gradient algorith, which computes the gradient of
          // adjoint and backward fields at each point in the element
          // The Lambda function is is passed to the gradient algorithm
          // which is applied to gradient result for every quadrature point
          specfem::algorithms::gradient(
              team, iterator, partial_derivatives,
              quadrature_element.hprime_gll, adjoint_element_field.displacement,
              backward_element_field.displacement,
              [&](const typename ChunkPolicy::iterator_type::index_type
                      &iterator_index,
                  const typename PointFieldDerivativesType::ViewType &df,
                  const typename PointFieldDerivativesType::ViewType &dg) {
                const auto index = iterator_index.index;
                // Load properties, adjoint field, and backward field
                // for the point
                // ------------------------------
                const auto point_properties = [&]() -> PointPropertiesType {
                  PointPropertiesType point_properties;
                  specfem::compute::load_on_device(index, properties,
                                                   point_properties);
                  return point_properties;
                }();

                const auto adjoint_point_field = [&]() {
                  AdjointPointFieldType adjoint_point_field;
                  specfem::compute::load_on_device(index, adjoint_field,
                                                   adjoint_point_field);
                  return adjoint_point_field;
                }();

                const auto backward_point_field = [&]() {
                  BackwardPointFieldType backward_point_field;
                  specfem::compute::load_on_device(index, backward_field,
                                                   backward_point_field);
                  return backward_point_field;
                }();
                // ------------------------------

                const PointFieldDerivativesType adjoint_point_derivatives(df);
                const PointFieldDerivativesType backward_point_derivatives(dg);

                // Compute the kernel for the point
                const auto point_kernel =
                    specfem::medium::compute_frechet_derivatives(
                        point_properties, adjoint_point_field,
                        backward_point_field, adjoint_point_derivatives,
                        backward_point_derivatives, dt);

                // Update the kernel in the global memory
                specfem::compute::add_on_device(index, point_kernel, kernels);
              });
        }
      });

  Kokkos::fence();
}
