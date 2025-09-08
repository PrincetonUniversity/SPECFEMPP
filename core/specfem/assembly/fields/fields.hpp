#pragma once

#include "data_access.hpp"
#include "dim2/simulation_field.hpp"
#include "dim3/simulation_field.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"

namespace specfem::assembly {
/**
 * @brief Store fields within the simulation
 *
 */
template <specfem::dimension::type DimensionTag> struct fields {
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag
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
  fields(const specfem::assembly::mesh<dimension_tag> &mesh,
         const specfem::assembly::element_types<dimension_tag> &element_types,
         const specfem::simulation::type simulation);
  ///@}

  /**
   * @brief Get the simulation field object
   *
   * @tparam ReturnFieldType Field type
   * @return specfem::assembly::simulation_field<fieldtype> Simulation field
   */
  template <specfem::wavefield::simulation_field ReturnFieldType>
  KOKKOS_INLINE_FUNCTION
      specfem::assembly::simulation_field<dimension_tag, ReturnFieldType>
      get_simulation_field() const {
    if constexpr (ReturnFieldType ==
                  specfem::wavefield::simulation_field::forward) {
      return forward;
    } else if constexpr (ReturnFieldType ==
                         specfem::wavefield::simulation_field::adjoint) {
      return adjoint;
    } else if constexpr (ReturnFieldType ==
                         specfem::wavefield::simulation_field::backward) {
      return backward;
    } else if constexpr (ReturnFieldType ==
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
  void copy_to_device();

  /**
   * @brief Copy fields to the host
   *
   */
  void copy_to_host();

  int nglob;
  specfem::assembly::simulation_field<
      dimension_tag,
      specfem::wavefield::simulation_field::buffer>
      buffer; ///< Buffer field. Generally used for temporary storage for
              ///< adjoint fields read from disk
  specfem::assembly::simulation_field<
      dimension_tag,
      specfem::wavefield::simulation_field::forward>
      forward; ///< Forward field
  specfem::assembly::simulation_field<
      dimension_tag,
      specfem::wavefield::simulation_field::adjoint>
      adjoint; ///< Adjoint field
  specfem::assembly::simulation_field<
      dimension_tag,
      specfem::wavefield::simulation_field::backward>
      backward; ///< Backward field
};

} // namespace specfem::assembly
