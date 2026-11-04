#pragma once

#include "fk_injection/utilities/complex_matrix.hpp"
#include <complex>

namespace fk_injection::impl {

/**
 * @brief Two-component acoustic state at a depth in a fluid layer.
 *
 * Carries the normal displacement @f$u_z@f$ and the scaled pressure
 * @f$P/k@f$ propagated by the acoustic 2x2 layer propagator.
 */
struct AcousticState {
  std::complex<double> u_z =
      std::complex<double>(0.0, 0.0); ///< Normal displacement @f$u_z@f$.
  std::complex<double> pressure_over_k =
      std::complex<double>(0.0, 0.0); ///< Scaled pressure @f$P/k@f$.
};

/**
 * @brief Acoustic state just inside the fluid at a fluid-solid interface.
 *
 * Applies the interface conditions (continuity of @f$u_z@f$ and of normal
 * traction @f$\sigma_{zz}@f$) to the elastic state propagated up to just below
 * the interface, returning the 2-component acoustic state. Faithful to
 * reference fk_core.cpp lines 488-490, without the unused solid slots.
 *
 * @param state Elastic state @f$(u_x, u_z, \sigma_{xz}, \sigma_{zz})@f$ just
 *              below the fluid-solid interface.
 * @return Acoustic state @f$(u_z, P/k)@f$ at the base of the fluid column.
 */
inline AcousticState
acoustic_from_solid(const fk_injection::ComplexVector<4> &state) {
  AcousticState result;
  result.u_z = state[1];              // normal displacement continuity
  result.pressure_over_k = -state[3]; // P/k = -sigma_zz
  return result;
}

} // namespace fk_injection::impl
