#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "specfem/tags.hpp"
#include "stiffness_kernels.hpp"
#include <Kokkos_Core.hpp>
#include <limits>

namespace specfem::compute::impl {

/**
 * @brief Compute stiffness interaction for a single element-tag combination.
 *
 * Dispatches the appropriate stiffness kernel (gather or scatter) for all
 * elements matching the compile-time tag combination in Tags.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation,
 *              boundary, wavefield, and mpi)
 * @param assembly The assembly object containing the mesh
 * @param istep Current time step
 * @return Number of elements processed
 */
template <int NGLL, typename Tags>
int compute_stiffness_interaction_core(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int &istep) {

  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto property_tag = Tags::property_tag;
  constexpr auto boundary_tag = Tags::boundary_tag;
  constexpr auto attenuation_tag = Tags::attenuation_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto mpi_tag = Tags::mpi_tag;

  const auto elements = assembly.element_types.get_elements_on_device(
      medium_tag, property_tag, attenuation_tag, boundary_tag, mpi_tag);

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return 0;

  if (assembly.mesh.element_grid != NGLL) {
    throw std::runtime_error(
        "The number of GLL points in the mesh elements must match "
        "the template parameter NGLL.");
  }

  specfem::mesh_entity::element_grid<dimension_tag,
                                     specfem::mesh_entity::Grid<NGLL>>
      element_grid{};

  using ParallelConfig = typename gather_kernel<NGLL, Tags>::ParallelConfig;

  specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), elements,
                                                  element_grid);

  Kokkos::Profiling::pushRegion("Compute Stiffness Interaction");

  if constexpr (Tags::boundary_tag == specfem::element::boundary_tag::stacey &&
                Tags::wavefield_tag ==
                    specfem::simulation::field_type::backward) {

    const auto &field =
        assembly.fields.template get_simulation_field<Tags::wavefield_tag>();
    const auto &boundary_values =
        assembly.boundary_values.template get_container<boundary_tag>();

    using PointAccelerationType =
        typename gather_kernel<NGLL, Tags>::PointAccelerationType;

    specfem::execution::for_all(
        "specfem::compute::compute_stiffness_interaction", chunk,
        KOKKOS_LAMBDA(
            const typename decltype(chunk)::base_index_type &iterator_index) {
          const auto index = iterator_index.get_index();
          PointAccelerationType acceleration;
          specfem::assembly::load_on_device(istep, index, boundary_values,
                                            acceleration);
          specfem::assembly::atomic_add_on_device(index, field, acceleration);
        });

  } else {

    // Gather kernel uses more shared memory but avoids atomics (~7% faster).
    // Scatter kernel uses less shared memory via atomic accumulation.
    // Select gather when it fits within the device limit;
    if (gather_kernel<NGLL, Tags>::shmem_size() <= chunk.scratch_size_max(0)) {
      gather_kernel<NGLL, Tags>{ assembly, istep }(chunk.set_scratch_size(
          0, Kokkos::PerTeam(gather_kernel<NGLL, Tags>::shmem_size())));
    } else if (scatter_kernel<NGLL, Tags>::shmem_size() <=
               chunk.scratch_size_max(0)) {
      scatter_kernel<NGLL, Tags>{ assembly, istep }(chunk.set_scratch_size(
          0, Kokkos::PerTeam(scatter_kernel<NGLL, Tags>::shmem_size())));
    } else if (gather_kernel<NGLL, Tags>::shmem_size() <=
               chunk.scratch_size_max(1)) {
      gather_kernel<NGLL, Tags>{ assembly, istep }(chunk.set_scratch_size(
          1, Kokkos::PerTeam(gather_kernel<NGLL, Tags>::shmem_size())));
    } else if (scatter_kernel<NGLL, Tags>::shmem_size() <=
               chunk.scratch_size_max(1)) {
      scatter_kernel<NGLL, Tags>{ assembly, istep }(chunk.set_scratch_size(
          1, Kokkos::PerTeam(scatter_kernel<NGLL, Tags>::shmem_size())));
    } else {
      throw std::runtime_error(
          "Not enough shared memory for stiffness interaction kernels.");
    }
  }

  Kokkos::Profiling::popRegion();

  return nelements;
}

/**
 * @brief Compute stiffness interaction with MPI communication-computation
 * overlap.
 *
 * Orchestrates the four-phase stiffness computation that overlaps MPI halo
 * exchange with inner-element work:
 * 1. Compute stiffness on outer elements (touching MPI boundaries)
 * 2. Begin async MPI exchange of acceleration
 * 3. Compute stiffness on inner elements (overlaps with MPI transfer)
 * 4. Complete MPI exchange and accumulate received contributions
 *
 * For dim2 (no MPI yet), all elements are classified as inner, the outer
 * pass finds zero elements, and begin/complete_exchange are no-ops.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags (dimension, medium, wavefield)
 * @param assembly The assembly object (mutated by MPI exchange)
 * @param istep Current time step
 * @return Number of elements updated across all tag combinations
 */
template <int NGLL, typename Tags>
int compute_stiffness_interaction(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int istep) {

  int elements_updated = 0;

  constexpr auto base_set =
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
      specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
      ATTENUATION_SET(none, constant_isotropic) *
      BOUNDARY_SET(none, stacey, acoustic_free_surface,
                   composite_stacey_dirichlet);
  constexpr auto outer_set = base_set * MPI_SET(outer);
  constexpr auto inner_set = base_set * MPI_SET(inner);

  auto compute_stiffness = [&]<typename ElementTags>() {
    elements_updated += compute_stiffness_interaction_core<
        NGLL, specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(assembly,
                                                                       istep);
  };

  // Phase 1: outer elements (touching MPI boundaries)
  specfem::tag_dispatch::for_each(outer_set, compute_stiffness);

  // Phase 2: begin async MPI exchange of acceleration
  auto &sim_field =
      assembly.fields.template get_simulation_field<Tags::wavefield_tag>();
  assembly.accel_buffers
      .template begin_exchange<Tags::wavefield_tag, Tags::medium_tag>(
          sim_field);

  // Phase 3: inner elements (overlaps with MPI transfer)
  specfem::tag_dispatch::for_each(inner_set, compute_stiffness);

  // Phase 4: complete MPI exchange and accumulate
  assembly.accel_buffers
      .template complete_exchange<Tags::wavefield_tag, Tags::medium_tag>(
          sim_field);

  return elements_updated;
}

} // namespace specfem::compute::impl
