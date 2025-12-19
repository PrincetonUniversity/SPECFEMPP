#pragma once

#include "data_access.hpp"
#include "dim2/simulation_field.hpp"
#include "dim3/simulation_field.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element simulation fields management container
 *
 * This class provides storage and management for all simulation fields
 * (wavefields) used in spectral element computations. It manages forward,
 * adjoint, backward, and buffer fields for wave propagation simulations,
 * seismic imaging, and full waveform inversion applications.
 *
 * The class handles different simulation field types:
 * - Forward fields: Primary wave propagation simulation
 * - Adjoint fields: Reverse-time adjoint wave propagation
 * - Backward fields: Backward propagation for gradient computation
 * - Buffer fields: Temporary storage for checkpointing and I/O
 *
 * Each field type contains the appropriate wavefield components based on
 * the medium type (displacement for elastic, potential for acoustic, etc.)
 * and spatial dimension. The class provides unified access to fields with
 * efficient device/host memory management.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 *
 * @code
 * // Construct fields for forward simulation
 * specfem::assembly::fields<specfem::dimension::type::dim2> sim_fields(
 *     mesh, element_types, specfem::simulation::type::forward);
 *
 * // Access forward wavefield
 * auto forward_field = sim_fields.get_simulation_field<
 *     specfem::wavefield::simulation_field::forward>();
 *
 * // Copy fields to device for computation
 * sim_fields.copy_to_device();
 * @endcode
 */
template <specfem::dimension::type DimensionTag> struct fields {
  constexpr static auto dimension_tag = DimensionTag;

  /**
   * @brief Default constructor.
   *
   * Initializes an empty fields container with no allocated storage.
   */
  fields() = default;

  /**
   * @brief Construct simulation fields from mesh and simulation configuration.
   *
   * Initializes all simulation fields (forward, adjoint, backward, buffer)
   * based on the mesh geometry, element types, and simulation requirements.
   * Allocates appropriate field storage for the specified spatial dimension
   * and medium types present in the mesh.
   *
   * @param mesh Assembly mesh containing global numbering and connectivity
   * @param element_types Element classification for field allocation
   * @param simulation Simulation type determining which fields to initialize
   *
   * @code
   * specfem::assembly::fields<specfem::dimension::type::dim2> fields(
   *     mesh, element_types, specfem::simulation::type::forward_adjoint);
   * @endcode
   */
  fields(const specfem::assembly::mesh<dimension_tag> &mesh,
         const specfem::assembly::element_types<dimension_tag> &element_types,
         const specfem::simulation::type simulation);

  /**
   * @brief Get simulation field of specified type.
   *
   * Template method that provides compile-time access to different simulation
   * field types. The return type is determined at compile time based on the
   * template parameter, enabling efficient field access in device kernels.
   *
   * @tparam ReturnFieldType Type of simulation field to retrieve
   * @return Reference to the requested simulation field
   *
   * @code
   * // Get forward field for wave propagation
   * auto forward = fields.get_simulation_field<
   *     specfem::wavefield::simulation_field::forward>();
   *
   * // Get adjoint field for reverse propagation
   * auto adjoint = fields.get_simulation_field<
   *     specfem::wavefield::simulation_field::adjoint>();
   * @endcode
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
   * @brief Copy all simulation fields from host to device memory.
   *
   * Transfers all active simulation fields (forward, adjoint, backward, buffer)
   * from host-accessible memory to device memory for GPU computations. This
   * is typically called before starting device-based wave propagation kernels.
   */
  void copy_to_device();

  /**
   * @brief Copy all simulation fields from device to host memory.
   *
   * Transfers all active simulation fields from device memory back to
   * host-accessible memory for post-processing, I/O operations, or debugging.
   * This is typically called after completing device-based computations.
   */
  void copy_to_host();

  int nglob; ///< Total number of global points in the mesh

  /**
   * @brief Buffer simulation field for temporary storage and checkpointing.
   *
   * Used for intermediate storage during time stepping, I/O operations,
   * and checkpointing in long simulations.
   */
  specfem::assembly::simulation_field<
      dimension_tag, specfem::wavefield::simulation_field::buffer>
      buffer;

  /**
   * @brief Forward simulation field for primary wave propagation.
   *
   * Contains displacement (elastic), potential (acoustic), or appropriate
   * field variables for forward-time wave propagation simulations.
   */
  specfem::assembly::simulation_field<
      dimension_tag, specfem::wavefield::simulation_field::forward>
      forward;

  /**
   * @brief Adjoint simulation field for reverse-time wave propagation.
   *
   * Used in adjoint methods for seismic imaging, full waveform inversion,
   * and gradient computation. Propagates backward in time from receivers.
   */
  specfem::assembly::simulation_field<
      dimension_tag, specfem::wavefield::simulation_field::adjoint>
      adjoint;

  /**
   * @brief Backward simulation field for gradient computation.
   *
   * Used in conjunction with adjoint fields for computing gradients in
   * full waveform inversion and optimization algorithms.
   */
  specfem::assembly::simulation_field<
      dimension_tag, specfem::wavefield::simulation_field::backward>
      backward;
};

} // namespace specfem::assembly
