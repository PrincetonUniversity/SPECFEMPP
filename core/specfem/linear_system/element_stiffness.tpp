#pragma once

#include "specfem/execution.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/impl/stiffness_probe_kernel.hpp"
#include "specfem/linear_system/impl/stiffness_tensor_graph_kernel.hpp"
#include "specfem/mesh_entity.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>

template <int NGLL, typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::linear_system::compute_element_stiffness(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::datatype::ElementIndexRange &batch,
    const Kokkos::View<type_real ***, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &k_e,
    const specfem::linear_system::StiffnessKernelImpl impl) {

  using KernelType =
      specfem::linear_system_impl::stiffness_probe_kernel<NGLL, Tags>;

  // Kernel-independent block edge length (every implementation honors it).
  constexpr int ndof = specfem::element::attributes<
                           Tags::dimension_tag, Tags::medium_tag>::components *
                       NGLL * NGLL * NGLL;

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
      static_cast<int>(k_e.extent(1)) != ndof ||
      static_cast<int>(k_e.extent(2)) != ndof) {
    throw std::runtime_error(
        "specfem::linear_system::compute_element_stiffness: the element "
        "stiffness buffer must have extents (>= batch size, ndof, ndof) "
        "with ndof = ncomp * NGLL^3.");
  }

  if (impl == specfem::linear_system::StiffnessKernelImpl::tensor_graph) {
    // Throws in builds without SPECFEM_ENABLE_TENSOROPS (see the impl header).
    specfem::linear_system_impl::compute_element_stiffness_tensor_graph<NGLL,
                                                                        Tags>(
        assembly, batch, k_e);
    return;
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

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::ElementStiffnessKernel
specfem::linear_system::make_element_stiffness_kernel(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const int batch_capacity,
    const specfem::linear_system::StiffnessKernelImpl impl) {

  // Runtime -> compile-time NGLL, mirroring the runtime dispatcher below
  // (only 5 is instantiated for 3D meshes).
  if (assembly.mesh.element_grid != 5) {
    throw std::runtime_error(
        "specfem::linear_system::make_element_stiffness_kernel: only "
        "NGLL == 5 is instantiated for 3D meshes.");
  }

  if (impl == specfem::linear_system::StiffnessKernelImpl::tensor_graph) {
    // Workspace allocation and identity fill happen here, once; shared_ptr
    // because std::function requires a copyable target.
    const auto kernel = std::make_shared<
        specfem::linear_system_impl::StiffnessTensorGraphKernel<5, Tags>>(
        assembly, batch_capacity);
    return [kernel](const specfem::datatype::ElementIndexRange &batch,
                    const auto &k_e) { (*kernel)(batch, k_e); };
  }

  // The probe kernel has no cross-batch state; delegate per call so its
  // validation and scratch sizing stay in one place.
  return [&assembly](const specfem::datatype::ElementIndexRange &batch,
                     const auto &k_e) {
    specfem::linear_system::compute_element_stiffness<5, Tags>(
        assembly, batch, k_e,
        specfem::linear_system::StiffnessKernelImpl::probe);
  };
}
