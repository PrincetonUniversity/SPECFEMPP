#pragma once

#include "specfem/execution.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/impl/stiffness_probe_kernel.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

template <int NGLL, typename Tags>
void specfem::linear_system::compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e) {

  static_assert(Tags::dimension_tag == specfem::element::dimension_tag::dim3,
                "compute_element_stiffness takes a dim3 assembly; Tags must "
                "be a dim3 bundle.");

  using KernelType =
      specfem::linear_system_impl::stiffness_probe_kernel<NGLL, Tags>;

  if (batch.empty()) {
    return;
  }

  if (assembly.mesh.element_grid != NGLL) {
    throw std::runtime_error(
        "specfem::linear_system::compute_element_stiffness: the number of "
        "GLL points in the mesh elements must match the template parameter "
        "NGLL.");
  }

  if (static_cast<int>(k_e.extent(0)) < batch.size() ||
      static_cast<int>(k_e.extent(1)) != KernelType::ndof ||
      static_cast<int>(k_e.extent(2)) != KernelType::ndof) {
    throw std::runtime_error(
        "specfem::linear_system::compute_element_stiffness: the element "
        "stiffness buffer must have extents (>= batch size, ndof, ndof) "
        "with ndof = ncomp * NGLL^3.");
  }

  specfem::mesh_entity::element_grid<specfem::element::dimension_tag::dim3,
                                     specfem::mesh_entity::Grid<NGLL>>
      element_grid{};

  using ParallelConfig = typename KernelType::ParallelConfig;

  specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), batch,
                                                  element_grid);

  KernelType kernel(assembly, batch.begin_index(), k_e);

  // No level-1 fallback: the chunk scratch types bind team_scratch(0) in
  // their constructors, so scratch requested at level 1 would never be used.
  if (KernelType::shmem_size() <= chunk.scratch_size_max(0)) {
    kernel(
        chunk.set_scratch_size(0, Kokkos::PerTeam(KernelType::shmem_size())));
  } else {
    throw std::runtime_error(
        "specfem::linear_system::compute_element_stiffness: not enough "
        "level-0 scratch memory for the stiffness probe kernel.");
  }

  Kokkos::fence();
}
