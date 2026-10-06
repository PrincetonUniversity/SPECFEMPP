#pragma once

#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Map a solid P-SV state vector to the acoustic state at a fluid/solid
 *        interface.
 *
 * Applies the interface coupling conditions — continuity of normal displacement
 * @f$u_z@f$ and continuity of normal traction @f$\sigma_{zz}@f$ — to convert
 * the 4-component elastic state @f$(u_x, u_z, \sigma_{xz}, \sigma_{zz})@f$
 * propagated up to just below the interface into the 2-component acoustic state
 * @f$(u_z, P/k)@f$ carried in the leading two slots of the returned vector.
 *
 * The mapping (faithful to reference fk_core.cpp lines 488-490):
 * @code
 *   result[0] = state[1];   // u_z: continuity of normal displacement
 *   result[1] = -state[3];  // P/k = -sigma_zz: continuity of normal traction
 *   result[2] = state[2];   // sigma_xz unchanged (unused in acoustic domain)
 *   result[3] = -state[3];  // duplicated for consistency with bot_vec layout
 * @endcode
 *
 * @param state  Elastic state vector @f$(u_x, u_z, \sigma_{xz}, \sigma_{zz})@f$
 *               at the bottom of the solid stack, just below the fluid/solid
 *               interface.
 * @return       State vector with @f$(u_z, P/k)@f$ in the leading two slots,
 *               ready for the acoustic propagator chain.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexVector<4>
couple_solid_to_fluid(specfem::utilities::ComplexVector<4> state) {
  specfem::utilities::ComplexVector<4> result;
  result[0] = state[1];  // u_z (normal displacement continuity)
  result[1] = -state[3]; // P/k = -sigma_zz (normal traction continuity)
  result[2] = state[2];  // sigma_xz carried unchanged
  result[3] = -state[3]; // P/k duplicated in slot 3 (bot_vec convention)
  return result;
}

} // namespace fk_impl
} // namespace injection
} // namespace specfem
