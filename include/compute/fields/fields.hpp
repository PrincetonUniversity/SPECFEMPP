#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/properties/interface.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "simulation_field.hpp"
#include "simulation_field.tpp"

namespace specfem {
namespace compute {
/**
 * @brief Store fields within the simulation
 *
 */
struct fields {

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  fields() = default;

  /**
   * @brief Contruct fields from an assembled mesh
   *
   * @param mesh Assembled mesh
   * @param properties Material properties
   * @param simulation Current simulation type
   */
  fields(const specfem::compute::mesh &mesh,
         const specfem::compute::element_types &element_types,
         const specfem::simulation::type simulation);
  ///@}

  /**
   * @brief Get the simulation field object
   *
   * @tparam fieldtype Field type
   * @return specfem::compute::simulation_field<fieldtype> Simulation field
   */
  template <specfem::wavefield::simulation_field fieldtype>
  KOKKOS_INLINE_FUNCTION specfem::compute::simulation_field<fieldtype>
  get_simulation_field() const {
    if constexpr (fieldtype == specfem::wavefield::simulation_field::forward) {
      return forward;
    } else if constexpr (fieldtype ==
                         specfem::wavefield::simulation_field::adjoint) {
      return adjoint;
    } else if constexpr (fieldtype ==
                         specfem::wavefield::simulation_field::backward) {
      return backward;
    } else if constexpr (fieldtype ==
                         specfem::wavefield::simulation_field::buffer) {
      return buffer;
    } else {
      static_assert("field type not supported");
    }
  }

  /**
   * @brief Copy fields to the device
   *
   */
  void copy_to_device() {
    buffer.copy_to_device();
    forward.copy_to_device();
    adjoint.copy_to_device();
    backward.copy_to_device();
  }

  /**
   * @brief Copy fields to the host
   *
   */
  void copy_to_host() {
    buffer.copy_to_host();
    forward.copy_to_host();
    adjoint.copy_to_host();
    backward.copy_to_host();
  }

  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::buffer>
      buffer; ///< Buffer field. Generally used for temporary storage for
              ///< adjoint fields read from disk
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::forward>
      forward; ///< Forward field
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::adjoint>
      adjoint; ///< Adjoint field
  specfem::compute::simulation_field<
      specfem::wavefield::simulation_field::backward>
      backward; ///< Backward field
};

} // namespace compute
} // namespace specfem
