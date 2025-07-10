#pragma once
#include "enumerations/macros.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem_setup.hpp"

namespace specfem::assembly::sources_impl {

// Forward declarations - must be before template definition
void compute_source_array(
    const std::shared_ptr<specfem::sources::adjoint_source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

void compute_source_array(
    const std::shared_ptr<specfem::sources::cosserat_force> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

void compute_source_array(
    const std::shared_ptr<specfem::sources::external> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

void compute_source_array(
    const std::shared_ptr<specfem::sources::force> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

void compute_source_array(
    const std::shared_ptr<specfem::sources::moment_tensor> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array);

/**
 * @brief Compute the source array for a given source
 *
 * This is a helper function that computes the source array for a given
 * source. We implement it here instead of the source to remove dependency of
 * source on the assembly module.
 */
template <specfem::dimension::type DimensionTag>
void compute_source_array(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::jacobian_matrix &jacobian_matrix,
    const specfem::assembly::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Check the type of the source and call the appropriate specialization
  if (std::shared_ptr<specfem::sources::adjoint_source> adjoint =
          std::dynamic_pointer_cast<specfem::sources::adjoint_source>(source)) {
    std::cout << "Computing source array for adjoint source." << std::endl;
    compute_source_array(adjoint, mesh, jacobian_matrix, element_types,
                         source_array);
    return;
  } else if (std::shared_ptr<specfem::sources::cosserat_force> cosserat =
                 std::dynamic_pointer_cast<specfem::sources::cosserat_force>(
                     source)) {
    std::cout << "Computing source array for Cosserat force source."
              << std::endl;
    compute_source_array(cosserat, mesh, jacobian_matrix, element_types,
                         source_array);
    return;
  } else if (std::shared_ptr<specfem::sources::external> external =
                 std::dynamic_pointer_cast<specfem::sources::external>(
                     source)) {
    std::cout << "Computing source array for external source." << std::endl;
    compute_source_array(external, mesh, jacobian_matrix, element_types,
                         source_array);
    return;
  } else if (std::shared_ptr<specfem::sources::force> force =
                 std::dynamic_pointer_cast<specfem::sources::force>(source)) {
    std::cout << "Force source x: " << force->get_x()
              << ", z: " << force->get_z() << ", angle: " << force->get_angle()
              << std::endl;
    // print force type
    compute_source_array(force, mesh, jacobian_matrix, element_types,
                         source_array);
    return;
  } else if (std::shared_ptr<specfem::sources::moment_tensor> moment_tensor =
                 std::dynamic_pointer_cast<specfem::sources::moment_tensor>(
                     source)) {
    std::cout << "Computing source array for moment tensor source."
              << std::endl;
    compute_source_array(moment_tensor, mesh, jacobian_matrix, element_types,
                         source_array);
    return;
  }
  // If the source type is not recognized, abort with an error message
  KOKKOS_ABORT_WITH_LOCATION(
      "For this source type, the compute_source_array "
      "function is not implemented. Please implement "
      "it in the source class or provide a specialization "
      "for this function.");
}

} // namespace specfem::assembly::sources_impl
