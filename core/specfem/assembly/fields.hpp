#pragma once

#include "element_types.hpp"
#include "enumerations/interface.hpp"
#include "fields/field_impl.hpp"
#include "fields/field_impl.tpp"
#include "fields/simulation_field.hpp"
#include "fields/simulation_field.tpp"
#include "mesh.hpp"

namespace specfem::assembly {
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
   * @param element_types Element types
   * @param simulation Current simulation type
   */
  fields(const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
         const specfem::assembly::element_types<specfem::dimension::type::dim2>
             &element_types,
         const specfem::simulation::type simulation);
  ///@}

  /**
   * @brief Get the simulation field object
   *
   * @tparam fieldtype Field type
   * @return specfem::assembly::simulation_field<fieldtype> Simulation field
   */
  template <specfem::wavefield::simulation_field fieldtype>
  KOKKOS_INLINE_FUNCTION specfem::assembly::simulation_field<fieldtype>
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

  specfem::assembly::simulation_field<
      specfem::wavefield::simulation_field::buffer>
      buffer; ///< Buffer field. Generally used for temporary storage for
              ///< adjoint fields read from disk
  specfem::assembly::simulation_field<
      specfem::wavefield::simulation_field::forward>
      forward; ///< Forward field
  specfem::assembly::simulation_field<
      specfem::wavefield::simulation_field::adjoint>
      adjoint; ///< Adjoint field
  specfem::assembly::simulation_field<
      specfem::wavefield::simulation_field::backward>
      backward; ///< Backward field
};

} // namespace specfem::assembly
