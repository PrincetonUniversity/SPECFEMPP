#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/impl/stiffness_kernel.hpp"
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
                       Kokkos::DefaultExecutionSpace> &k_e) {

  using KernelType = specfem::linear_system_impl::StiffnessKernel<NGLL, Tags>;

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

  // Throws in builds without SPECFEM_ENABLE_TENSOROPS (see the impl header).
  const KernelType kernel(assembly);
  kernel(batch, k_e);
  Kokkos::fence();
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::ElementStiffnessKernel
specfem::linear_system::make_element_stiffness_kernel(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {

  // Runtime -> compile-time NGLL, mirroring the runtime dispatcher below
  // (only 5 is instantiated for 3D meshes).
  if (assembly.mesh.element_grid != 5) {
    throw std::runtime_error(
        "specfem::linear_system::make_element_stiffness_kernel: only "
        "NGLL == 5 is instantiated for 3D meshes.");
  }

  // Stateless (the graph lives in team scratch), but bound once so the
  // mesh-grid validation runs here rather than per batch. shared_ptr because
  // std::function requires a copyable target. Throws in builds without
  // SPECFEM_ENABLE_TENSOROPS.
  const auto kernel =
      std::make_shared<specfem::linear_system_impl::StiffnessKernel<5, Tags>>(
          assembly);
  return [kernel](const specfem::datatype::ElementIndexRange &batch,
                  const auto &k_e) { (*kernel)(batch, k_e); };
}
