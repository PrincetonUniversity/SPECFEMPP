#pragma once

#include "specfem/boundary_conditions.hpp"
#include "specfem/assembly.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/execution.hpp"
#include "specfem/point.hpp"
#include "compute_source_interaction.hpp"
#include <Kokkos_Core.hpp>

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_source_interaction(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly, const int &timestep) {

  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto property_tag = Tags::property_tag;
  constexpr auto boundary_tag = Tags::boundary_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto wavefield = Tags::wavefield_tag;

  const auto [element_indices, source_indices] =
      assembly.sources.get_sources_on_device(medium_tag, property_tag,
                                             boundary_tag, wavefield);

  // Get the element grid (ngllx, ngllz)
  const auto &element_grid = assembly.mesh.element_grid;

  // Check if the number of GLL points in the mesh elements matches the template
  // parameter NGLL
  if (element_grid != NGLL) {
    throw std::runtime_error("The number of GLL points in the mesh elements must match "
                             "the template parameter NGLL.");
  }

  auto &sources = assembly.sources;

  const int nsources = source_indices.extent(0);

  if (nsources == 0)
    return;

  // Some aliases
  const auto &properties = assembly.properties;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  sources.update_timestep(timestep);

  using PointSourceType =
      specfem::point::source<dimension_tag, medium_tag, wavefield>;
  using PointPropertiesType =
      specfem::point::properties<dimension_tag, medium_tag, property_tag, false>;
  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, false>;
  using PointIndexType = specfem::point::mapped_index<dimension_tag, false>;

  using simd = specfem::datatype::simd<type_real, false>;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr int nthreads = 32;
  constexpr int lane_size = 1;
#else
  constexpr int nthreads = 1;
  constexpr int lane_size = 1;
#endif

  using ParallelConfig =
      specfem::parallel_configuration::chunk_config<dimension_tag, 1, 1, nthreads,
                                             lane_size, simd,
                                             Kokkos::DefaultExecutionSpace>;

  specfem::execution::MappedChunkedDomainIterator mapped_policy(
      ParallelConfig(), element_indices, source_indices, element_grid);

  Kokkos::Profiling::pushRegion("Compute Source Interaction");

  specfem::execution::for_all(
      "specfem::compute::compute_source_interaction", mapped_policy,
      KOKKOS_LAMBDA(const typename decltype(mapped_policy)::base_index_type &iterator_index) {
        const auto mapped_index = iterator_index.get_index();
        PointSourceType point_source;
        specfem::assembly::load_on_device(mapped_index, sources, point_source);

        PointPropertiesType point_property;
        specfem::assembly::load_on_device(mapped_index, properties,
                                         point_property);

        auto acceleration = specfem::medium_physics::compute_source_contribution(
            point_source, point_property);

        specfem::assembly::atomic_add_on_device(mapped_index, field, acceleration);
      });

  Kokkos::Profiling::popRegion();
}
