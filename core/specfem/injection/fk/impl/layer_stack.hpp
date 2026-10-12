#pragma once

#include "specfem/injection/fk/impl/fk_setup.hpp"
#include "specfem/injection/fk/impl/medium_interface_coupler.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <vector>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Physical state vectors at every layer interface, per frequency.
 *
 * Replaces the cumulative propagator-matrix chains of the old driver with the
 * physically meaningful interface states. For each positive frequency f:
 * - @c modes(f) is the bottom mode vector b(omega);
 * - @c elastic_top(f, L) is the 4-vector state at the TOP of elastic layer L
 *   (= bottom of layer L-1); L = n_elastic-1 is the half-space top = E*b;
 * - @c acoustic_bottom(f, j) is the 2-component state at the BOTTOM of fluid
 *   layer j (empty when the model has no fluid).
 *
 * Host-only intermediate (flat std::vector storage); consumed by depth_series
 * via element access, never handed to a raw-pointer transform.
 */
struct InterfaceStates {
  int number_of_frequencies = 0;
  int number_of_elastic_layers = 0;
  int number_of_fluid_layers = 0;
  std::vector<specfem::utilities::ComplexVector<4>> modes;
  std::vector<specfem::utilities::ComplexVector<4>> elastic_top_storage;
  std::vector<AcousticState> acoustic_bottom_storage;

  /// @brief Bottom mode vector b(omega) at frequency @p f.
  const specfem::utilities::ComplexVector<4> &modes_at(int f) const {
    return modes[f];
  }
  /// @brief State at the top of elastic layer @p layer, frequency @p f.
  const specfem::utilities::ComplexVector<4> &elastic_top(int f,
                                                          int layer) const {
    return elastic_top_storage[f * number_of_elastic_layers + layer];
  }
  /// @brief Acoustic state at the bottom of fluid layer @p layer, frequency @p
  /// f.
  const AcousticState &acoustic_bottom(int f, int layer) const {
    return acoustic_bottom_storage[f * number_of_fluid_layers + layer];
  }
};

/**
 * @brief Solve the layered elastic/acoustic stack for all stored frequencies.
 *
 * For each positive frequency: assemble the propagated elastic matrix N and the
 * full fluid propagator, solve the free-surface condition for the half-space
 * coefficients, build the bottom mode vector, then walk the interface states up
 * through the elastic layers and (if present) the fluid column.
 *
 * @note Assumes the model is a solid half-space with a contiguous elastic block
 *       and an optional fluid block on top of it (the @c LayeredModel
 *       invariant). Only a single solid-to-fluid coupling is applied, at the
 *       top of the elastic block. Generalizing to interleaved layering (e.g.
 *       solid-fluid-solid) is localized to this stage plus @c LayeredModel /
 *       @c locate: replace the two fixed elastic/fluid walks with one bottom-up
 *       per-interface walk that applies a same-medium propagator or the
 *       appropriate medium coupling at each interface, and add the companion
 *       fluid-to-solid coupling next to @c couple_solid_to_fluid. A fluid base
 *       would additionally need a non-elastic half-space eigen-decomposition.
 *       Downstream stages consume only medium-agnostic interface states and are
 *       unaffected.
 *
 * @param model Layered velocity model.
 * @param setup Shared frequency-domain setup (E matrix, omega, amplitudes).
 * @return Interface states for every stored frequency.
 */
InterfaceStates
solve_layer_stack(const specfem::injection::fk::LayeredModel &model,
                  const FkSetup &setup);

} // namespace fk_impl
} // namespace injection
} // namespace specfem
