#include "impl/layer_stack.hpp"
#include "plane_wave_response.hpp"

#include "impl/fk_math_impl.hpp"
#include "impl/halfspace_solve.hpp"
#include "impl/layer_operators.hpp"
#include <array>
#include <complex>
#include <variant>

fk_injection::impl::InterfaceStates fk_injection::impl::solve_layer_stack(
    const fk_injection::LayeredModel &model,
    const fk_injection::impl::FkSetup &setup) {
  using Cplx = std::complex<double>;

  const auto &layers = model.layers();
  const int number_of_layers = model.number_of_layers();
  const int halfspace = number_of_layers - 1;
  const bool has_fluid =
      std::holds_alternative<fk_injection::AcousticLayer>(layers.front());
  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const double p = setup.p;
  const fk_injection::incident_wave_type type = setup.wave_type;

  InterfaceStates states;
  states.number_of_frequencies = nf2;
  states.number_of_layers = number_of_layers;
  states.modes.resize(nf2);
  states.top_storage.resize(static_cast<std::size_t>(nf2) * number_of_layers);
  states.bottom_storage.resize(static_cast<std::size_t>(nf2) * halfspace);

  const fk_injection::ComplexMatrix<4> &e_mat = setup.e_mat;

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    // N = (cumulative elastic propagator) * E and the full fluid propagator
    // Q_0 * Q_1 * ... * Q_{n_fluid-1}, for the free-surface solve.
    fk_injection::ComplexMatrix<4> n_mat = e_mat;
    fk_injection::ComplexMatrix<2> fluid_propagator(
        [](int i, int j) { return i == j ? 1.0 : 0.0; });
    for (int layer = halfspace - 1; layer >= 0; --layer) {
      if (const auto *solid = std::get_if<fk_injection::ElasticIsotropicLayer>(
              &layers[layer])) {
        n_mat = fk_injection::impl::layer_propagator(*solid, omega,
                                                     solid->thickness, p) *
                n_mat;
      }
    }
    for (int layer = 0; layer < halfspace; ++layer) {
      if (const auto *fluid =
              std::get_if<fk_injection::AcousticLayer>(&layers[layer])) {
        fluid_propagator =
            fluid_propagator * fk_injection::impl::layer_propagator(
                                   *fluid, omega, fluid->thickness, p);
      }
    }

    const std::array<Cplx, 2> coeffs =
        has_fluid ? fk_injection::impl::halfspace_coefficients(
                        n_mat, fluid_propagator, type, setup.amplitude)
                  : fk_injection::impl::halfspace_coefficients(n_mat, type,
                                                               setup.amplitude);
    const fk_injection::ComplexVector<4> modes =
        fk_injection::impl::build_bottom_vector(type, setup.amplitude, coeffs);
    states.modes[f] = modes;

    // Walk up from the half-space top (E*b): a layer's bottom state is the
    // top state of the layer below, coupled into its medium.
    InterfaceState *top = states.top_storage.data() + f * number_of_layers;
    InterfaceState *bottom = states.bottom_storage.data() + f * halfspace;
    top[halfspace] = e_mat * modes;
    for (int layer = halfspace - 1; layer >= 0; --layer) {
      if (const auto *solid = std::get_if<fk_injection::ElasticIsotropicLayer>(
              &layers[layer])) {
        bottom[layer] =
            std::get<fk_injection::ComplexVector<4>>(top[layer + 1]);
        top[layer] = fk_injection::impl::elastic_state_in_layer(
            *solid, omega, solid->thickness, p,
            std::get<fk_injection::ComplexVector<4>>(bottom[layer]));
      } else {
        const auto &fluid =
            std::get<fk_injection::AcousticLayer>(layers[layer]);
        bottom[layer] =
            std::holds_alternative<fk_injection::ComplexVector<4>>(
                top[layer + 1])
                ? fk_injection::impl::acoustic_from_solid(
                      std::get<fk_injection::ComplexVector<4>>(top[layer + 1]))
                : std::get<fk_injection::impl::AcousticState>(top[layer + 1]);
        top[layer] = fk_injection::impl::acoustic_state_in_layer(
            fluid, omega, fluid.thickness, p,
            std::get<fk_injection::impl::AcousticState>(bottom[layer]));
      }
    }
  }

  return states;
}
