#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "stiffness_kernels.hpp"
#include <Kokkos_Core.hpp>
#include <limits>

namespace specfem::compute::impl {

template <int NGLL, typename Tags>
int compute_stiffness_interaction(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int &istep) {

  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto property_tag = Tags::property_tag;
  constexpr auto boundary_tag = Tags::boundary_tag;
  constexpr auto attenuation_tag = Tags::attenuation_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;

  const auto elements = assembly.element_types.get_elements_on_device(
      medium_tag, property_tag, attenuation_tag, boundary_tag);

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

  if constexpr ((Tags::boundary_tag == specfem::element::boundary_tag::stacey ||
                 Tags::boundary_tag == specfem::element::boundary_tag::
                                           composite_stacey_dirichlet) &&
                Tags::wavefield_tag ==
                    specfem::simulation::field_type::backward) {

    const auto &field =
        assembly.fields.template get_simulation_field<Tags::wavefield_tag>();
    const auto &boundary_values =
        assembly.boundary_values.template get_container<boundary_tag>();

    using PointAccelerationType =
        typename gather_kernel<NGLL, Tags>::PointAccelerationType;

    if (istep > 0) {
      const int boundary_istep = istep - 1;
      specfem::execution::for_all(
          "specfem::compute::compute_stiffness_interaction", chunk,
          KOKKOS_LAMBDA(
              const typename decltype(chunk)::base_index_type &iterator_index) {
            const auto index = iterator_index.get_index();
            PointAccelerationType acceleration;
            specfem::assembly::load_on_device(boundary_istep, index,
                                              boundary_values, acceleration);
            specfem::assembly::atomic_add_on_device(index, field, acceleration);
          });
    }

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

} // namespace specfem::compute::impl
