#pragma once

#include "specfem/medium/dim2/acoustic/isotropic/frechet_derivative.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/frechet_derivative.hpp"
#include "specfem/medium/dim2/elastic/isotropic/frechet_derivative.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/frechet_derivative.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @brief Compute Fréchet derivatives for seismic inversion kernels.
 *
 * Calculates sensitivity kernels used in adjoint-based seismic inversion
 * by computing derivatives of wavefield observables with respect to
 * material parameters. Dispatches to medium-specific implementations
 * based on dimension, medium type, and property tags encoded in Tags.
 *
 * @tparam Tags Compile-time tag bundle encoding dimension, medium, property,
 *              and SIMD configuration
 *
 * @param properties Material properties (density, elastic moduli)
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Sensitivity kernels for material parameters
 *
 * @code
 * auto kernels = compute_frechet_derivatives<Tags>(
 *     properties, adjoint_vel, adjoint_acc,
 *     backward_disp, adj_deriv, back_deriv, dt);
 * @endcode
 */
template <typename Tags>
KOKKOS_INLINE_FUNCTION auto compute_frechet_derivatives(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::velocity<Tags> &adjoint_velocity,
    const specfem::point::acceleration<Tags> &adjoint_acceleration,
    const specfem::point::displacement<Tags> &backward_displacement,
    const specfem::point::field_derivatives<Tags> &adjoint_derivatives,
    const specfem::point::field_derivatives<Tags> &backward_derivatives,
    const type_real &dt) {

  return specfem::medium_physics::impl_compute_frechet_derivatives<Tags>(
      properties, adjoint_velocity, adjoint_acceleration, backward_displacement,
      adjoint_derivatives, backward_derivatives, dt);
}

} // namespace medium_physics
} // namespace specfem
