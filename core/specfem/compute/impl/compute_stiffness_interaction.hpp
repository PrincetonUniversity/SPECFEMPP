#pragma once

#include "specfem/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/tags.hpp"

namespace specfem {
namespace compute {
namespace impl {

/**
 * @brief Computes stiffness matrix-vector product for spectral element wave
 * propagation.
 *
 * Calculates elastic forces from displacement field gradients, applies
 * stress-strain relations, computes divergence to get acceleration, and
 * enforces boundary conditions. Core computational kernel for seismic wave
 * simulation time stepping.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam WavefieldType Simulation field type (forward/backward/adjoint)
 * @tparam NGLL Number of GLL points per element dimension
 * @tparam MediumTag Medium type (elastic/acoustic/poroelastic)
 * @tparam PropertyTag Material properties (isotropic/anisotropic)
 * @tparam BoundaryTag Boundary conditions (none/free_surface/absorbing)
 *
 * @param assembly Complete spectral element assembly with mesh, fields,
 * properties
 * @param istep Current time step for boundary value storage
 *
 * @return Number of processed elements matching template parameters
 */
template <int NGLL, typename Tags>
int compute_stiffness_interaction(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const int &istep);
} // namespace impl
} // namespace compute
} // namespace specfem
