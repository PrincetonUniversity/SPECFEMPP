#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "boundary_conditions/boundary_conditions.tpp"
#include "specfem/assembly.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_source.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "execution/mapped_chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_source_interaction(
    specfem::assembly::assembly<DimensionTag> &assembly, const int &timestep) {

  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;
  constexpr auto dimension = DimensionTag;
  constexpr auto wavefield = WavefieldType;

  const auto [element_indices, source_indices] =
      assembly.sources.get_sources_on_device(MediumTag, PropertyTag,
                                             BoundaryTag, WavefieldType);

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
      specfem::point::source<dimension, medium_tag, wavefield>;
  using PointPropertiesType =
      specfem::point::properties<dimension, medium_tag, property_tag, false>;
  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, false>;
  using PointIndexType = specfem::point::mapped_index<dimension, false>;

      using simd = specfem::datatype::simd<type_real, false>;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr int nthreads = 32;
  constexpr int lane_size = 1;
#else
  constexpr int nthreads = 1;
  constexpr int lane_size = 1;
#endif

  using ParallelConfig =
      specfem::parallel_config::chunk_config<dimension, 1, 1, nthreads,
                                             lane_size, simd,
                                             Kokkos::DefaultExecutionSpace>;

  specfem::execution::MappedChunkedDomainIterator mapped_policy(
      ParallelConfig(), element_indices, source_indices, element_grid);

  Kokkos::Profiling::pushRegion("Compute Source Interaction");

  specfem::execution::for_all(
      "specfem::kokkos_kernels::compute_source_interaction", mapped_policy,
      KOKKOS_LAMBDA(const PointIndexType &mapped_index) {
        PointSourceType point_source;
        specfem::assembly::load_on_device(mapped_index, sources, point_source);

        PointPropertiesType point_property;
        specfem::assembly::load_on_device(mapped_index, properties,
                                         point_property);

        auto acceleration = specfem::medium::compute_source_contribution(
            point_source, point_property);

        specfem::assembly::atomic_add_on_device(mapped_index, field, acceleration);
      });

  Kokkos::Profiling::popRegion();
}
