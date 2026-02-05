#pragma once

#include "enumerations/interface_tags.hpp"
#include "specfem/assembly.hpp"
#include <type_traits>

namespace specfem::compute::impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::simulation::field_type WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::interface::flux_scheme_tag FluxSchemeTag>
void compute_coupling_weakly_conforming(
    const specfem::assembly::assembly<DimensionTag> &assembly);

template <specfem::element::dimension_tag DimensionTag,
          specfem::simulation::field_type WavefieldType, int NGLL,
          int NQuad_interface, specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::interface::flux_scheme_tag FluxSchemeTag>
void compute_coupling_nonconforming(
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
template <specfem::element::dimension_tag DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::simulation::field_type WavefieldType, int NGLL,
          int NQuad_intersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::interface::flux_scheme_tag FluxSchemeTag>
void compute_coupling(
    const specfem::assembly::assembly<DimensionTag> &assembly) {
  // Create dispatch tag for connection type
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, ConnectionTag>;

  // Forward to implementation with dispatch tag
  if constexpr (ConnectionTag == specfem::connections::type::nonconforming) {
    compute_coupling_nonconforming<DimensionTag, WavefieldType, NGLL,
                                   NQuad_intersection, InterfaceTag,
                                   BoundaryTag, FluxSchemeTag>(assembly);
  } else {
    compute_coupling_weakly_conforming<
        DimensionTag, WavefieldType, InterfaceTag, BoundaryTag, FluxSchemeTag>(
        assembly);
  }
}
} // namespace specfem::compute::impl
