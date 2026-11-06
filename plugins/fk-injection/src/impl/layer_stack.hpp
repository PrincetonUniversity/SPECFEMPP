#pragma once

#include "fk_injection/layered_model.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/field_recovery.hpp"
#include <variant>
#include <vector>

namespace fk_injection::impl {

struct FkSetup; // defined with PlaneWaveResponse

/// @brief Field state at an interface, in the medium of the layer it bounds:
/// the 4-vector in a solid, the 2-vector in a fluid.
using InterfaceState = std::variant<fk_injection::impl::AcousticState,
                                    fk_injection::ComplexVector<4>>;

/**
 * @brief Field state at the top and bottom of every layer, per frequency.
 *
 * Layers are indexed as in @c LayeredModel::layers. The half-space has a top
 * state, @f$E b@f$, but no bottom one: its field is given by the mode vector
 * @f$b(\omega)@f$.
 */
struct InterfaceStates {
  int number_of_frequencies = 0;
  int number_of_layers = 0;
  std::vector<fk_injection::ComplexVector<4>> modes;
  std::vector<InterfaceState> top_storage;    ///< (frequency, layer)
  std::vector<InterfaceState> bottom_storage; ///< (frequency, finite layer)

  /// @brief Bottom mode vector b(omega) at frequency @p f.
  const fk_injection::ComplexVector<4> &modes_at(int f) const {
    return modes[f];
  }
  /// @brief State at the top of @p layer, frequency @p f.
  const InterfaceState &top(int f, int layer) const {
    return top_storage[f * number_of_layers + layer];
  }
  /// @brief State at the bottom of finite @p layer, frequency @p f.
  const InterfaceState &bottom(int f, int layer) const {
    return bottom_storage[f * (number_of_layers - 1) + layer];
  }
};

/**
 * @brief Solve the layered elastic/acoustic stack for all stored frequencies.
 *
 * For each positive frequency: assemble the propagated elastic matrix N and the
 * full fluid propagator, solve the free-surface condition for the half-space
 * coefficients, build the bottom mode vector, then walk the interface states up
 * from the half-space top, coupling solid to fluid where the medium changes.
 *
 * @note The only stage that assumes the @c LayeredModel layering, a solid
 *       block with an optional fluid block on top: the free-surface solve
 *       takes one elastic and one fluid propagator. Interleaved layering (e.g.
 *       solid-fluid-solid) needs a per-interface solve here plus the
 *       fluid-to-solid coupling next to @c acoustic_from_solid in the walk. A
 *       fluid base would additionally need a non-elastic half-space
 *       eigen-decomposition.
 *
 * @param model Layered velocity model.
 * @param setup Shared frequency-domain setup (E matrix, omega, amplitudes).
 * @return Interface states for every stored frequency.
 */
InterfaceStates solve_layer_stack(const fk_injection::LayeredModel &model,
                                  const FkSetup &setup);

} // namespace fk_injection::impl
