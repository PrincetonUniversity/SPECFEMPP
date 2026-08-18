#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include <type_traits>

namespace specfem::compute::impl {

template <int NGLL, int NQuad_intersection, typename Tags>
void coupling_core_weakly_conforming(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly);

template <int NGLL, int NQuad_intersection, typename Tags>
void coupling_core_nonconforming(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly);

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
template <int NGLL, typename Tags>
void compute_coupling(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly);
} // namespace specfem::compute::impl
