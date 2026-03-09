#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace compute {
namespace impl {

/**
 * @brief Compute the mass matrix for the given medium.
 *
 * This function computes the mass matrix for the specified medium type and
 * properties. It is specialized for different dimension tags, wavefield types,
 * medium tags, property tags, and boundary tags.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam WavefieldType Simulation wavefield type (e.g., forward, adjoint,
 * backward)
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium type (e.g., elastic, acoustic)
 * @tparam PropertyTag Material property type (e.g., isotropic, anisotropic)
 * @tparam BoundaryTag Boundary condition type (e.g., free_surface, absorbing)
 *
 * @param dt Time step size (used for time-dependent mass matrix computations)
 * @param assembly SPECFEM++ assembly object.
 */
template <int NGLL, typename Tags>
void compute_mass_matrix(
    const type_real &dt,
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly);
} // namespace impl
} // namespace compute
} // namespace specfem
