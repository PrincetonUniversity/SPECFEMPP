#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include <type_traits>

namespace specfem::kokkos_kernels::impl {

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly);

/**
 * @brief Compute coupling between different media.
 *
 * This function computes the coupling between different media specified by the
 * template parameters.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam ConnectionTag Interface connection type to consider
 * (weakly_conforming/nonconforming)
 * @tparam WavefieldType Simulation wavefield type (e.g., forward, adjoint,
 * backward)
 * @tparam InterfaceTag Interface coupling type
 * (elastic_acoustic/acoustic_elastic)
 * @tparam BoundaryTag Boundary condition type (e.g., free_surface, absorbing)
 *
 * @param assembly SPECFEM++ assembly object.
 */
template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void compute_coupling(
    const specfem::assembly::assembly<DimensionTag> &assembly) {
  // Create dispatch tag for connection type
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, ConnectionTag>;

  // Forward to implementation with dispatch tag
  compute_coupling<DimensionTag, WavefieldType, InterfaceTag, BoundaryTag>(
      connection_dispatch(), assembly);
}
} // namespace specfem::kokkos_kernels::impl
