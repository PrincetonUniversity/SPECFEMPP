#include "impl/layer_stack.hpp"

#include "impl/fk_math_impl.hpp"
#include "impl/halfspace_solve.hpp"
#include "impl/layer_operators.hpp"
#include <array>
#include <complex>

fk_injection::impl::InterfaceStates fk_injection::impl::solve_layer_stack(
    const fk_injection::LayeredModel &model,
    const fk_injection::impl::FkSetup &setup) {
  using Cplx = std::complex<double>;

  const auto &elastic = model.elastic_layers();
  const auto &fluid = model.fluid_layers();
  const int n_elastic = model.number_of_elastic_layers();
  const int n_fluid = model.number_of_fluid_layers();
  const bool has_fluid = model.has_fluid_layer();
  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const double p = setup.p;
  const fk_injection::incident_wave_type type = setup.wave_type;

  InterfaceStates states;
  states.number_of_frequencies = nf2;
  states.number_of_elastic_layers = n_elastic;
  states.number_of_fluid_layers = n_fluid;
  states.modes.resize(nf2);
  states.elastic_top_storage.resize(static_cast<std::size_t>(nf2) * n_elastic);
  if (has_fluid) {
    states.acoustic_bottom_storage.resize(static_cast<std::size_t>(nf2) *
                                          n_fluid);
  }

  const fk_injection::ComplexMatrix<4> &e_mat = setup.e_mat;

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    // N = (cumulative elastic propagator) * E, built bottom-up over the finite
    // layers (L = n_elastic-2 .. 0).
    fk_injection::ComplexMatrix<4> n_mat = e_mat;
    for (int layer = n_elastic - 2; layer >= 0; --layer) {
      n_mat = fk_injection::impl::layer_propagator(
                  elastic[layer], omega, elastic[layer].thickness, p) *
              n_mat;
    }

    // Full fluid propagator Q_0 * Q_1 * ... * Q_{n_fluid-1}.
    fk_injection::ComplexMatrix<2> fluid_propagator(
        [](int i, int j) { return i == j ? 1.0 : 0.0; });
    for (int layer = 0; layer < n_fluid; ++layer) {
      fluid_propagator = fluid_propagator *
                         fk_injection::impl::layer_propagator(
                             fluid[layer], omega, fluid[layer].thickness, p);
    }

    const std::array<Cplx, 2> coeffs =
        has_fluid ? fk_injection::impl::halfspace_coefficients(
                        n_mat, fluid_propagator, type, setup.amplitude)
                  : fk_injection::impl::halfspace_coefficients(n_mat, type,
                                                               setup.amplitude);
    const fk_injection::ComplexVector<4> modes =
        fk_injection::impl::build_bottom_vector(type, setup.amplitude, coeffs);
    states.modes[f] = modes;

    // Elastic interface states, walking up from the half-space top (E*b).
    states.elastic_top_storage[f * n_elastic + (n_elastic - 1)] = e_mat * modes;
    for (int layer = n_elastic - 2; layer >= 0; --layer) {
      states.elastic_top_storage[f * n_elastic + layer] =
          fk_injection::impl::elastic_state_in_layer(
              elastic[layer], omega, elastic[layer].thickness, p,
              states.elastic_top_storage[f * n_elastic + (layer + 1)]);
    }

    // Acoustic interface states: couple the top-of-elastic state into the
    // fluid, then walk up through the fluid column.
    if (has_fluid) {
      states.acoustic_bottom_storage[f * n_fluid + (n_fluid - 1)] =
          fk_injection::impl::acoustic_from_solid(
              states.elastic_top_storage[f * n_elastic + 0]);
      for (int layer = n_fluid - 2; layer >= 0; --layer) {
        states.acoustic_bottom_storage[f * n_fluid + layer] =
            fk_injection::impl::acoustic_state_in_layer(
                fluid[layer + 1], omega, fluid[layer + 1].thickness, p,
                states.acoustic_bottom_storage[f * n_fluid + (layer + 1)]);
      }
    }
  }

  return states;
}
