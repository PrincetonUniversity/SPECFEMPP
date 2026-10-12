#include "specfem/injection/fk/impl/layer_stack.hpp"

#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/halfspace_solve.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"

#include <Kokkos_Complex.hpp>

specfem::injection::fk_impl::InterfaceStates
specfem::injection::fk_impl::solve_layer_stack(
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk_impl::FkSetup &setup) {
  using Cplx = Kokkos::complex<double>;

  const auto &elastic = model.elastic_layers();
  const auto &fluid = model.fluid_layers();
  const int n_elastic = model.number_of_elastic_layers();
  const int n_fluid = model.number_of_fluid_layers();
  const bool has_fluid = model.has_fluid_layer();
  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const type_real p = static_cast<type_real>(setup.p);
  const specfem::injection::fk::incident_wave_type type = setup.wave_type;

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

  const specfem::utilities::ComplexMatrix<4> &e_mat = setup.e_mat;
  const Cplx c1 = setup.amplitude.c1;
  const Cplx c3 = setup.amplitude.c3;

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    // N = (cumulative elastic propagator) * E, built bottom-up over the finite
    // layers (L = n_elastic-2 .. 0).
    specfem::utilities::ComplexMatrix<4> n_mat = e_mat;
    for (int layer = n_elastic - 2; layer >= 0; --layer) {
      const double g0 =
          specfem::injection::fk_impl::elastic_gamma0(elastic[layer], p);
      const specfem::utilities::ComplexMatrix<4> propagator =
          specfem::injection::fk_impl::layer_propagator(
              elastic[layer], omega, elastic[layer].thickness, p);
      n_mat = propagator * n_mat;
      n_mat *= Cplx(g0, 0.0);
    }

    // Full fluid propagator Q_0 * Q_1 * ... * Q_{n_fluid-1}.
    specfem::utilities::ComplexMatrix<4> fluid_propagator =
        specfem::utilities::ComplexMatrix<4>::identity();
    if (has_fluid) {
      for (int layer = 0; layer < n_fluid; ++layer) {
        const specfem::utilities::ComplexMatrix<4> q =
            specfem::injection::fk_impl::layer_propagator(
                fluid[layer], omega, fluid[layer].thickness, p);
        fluid_propagator = fluid_propagator * q;
      }
    }

    const Kokkos::Array<Cplx, 2> coeffs =
        specfem::injection::fk_impl::halfspace_coefficients(
            n_mat, has_fluid, fluid_propagator, type, c1, c3);
    const specfem::utilities::ComplexVector<4> modes =
        specfem::injection::fk_impl::build_bottom_vector(type, c1, c3, coeffs);
    states.modes[f] = modes;

    // Elastic interface states, walking up from the half-space top (E*b).
    states.elastic_top_storage[f * n_elastic + (n_elastic - 1)] = e_mat * modes;
    for (int layer = n_elastic - 2; layer >= 0; --layer) {
      states.elastic_top_storage[f * n_elastic + layer] =
          specfem::injection::fk_impl::elastic_state_in_layer(
              elastic[layer], omega, elastic[layer].thickness, p,
              states.elastic_top_storage[f * n_elastic + (layer + 1)]);
    }

    // Acoustic interface states: couple the top-of-elastic state into the
    // fluid, then walk up through the fluid column.
    if (has_fluid) {
      states.acoustic_bottom_storage[f * n_fluid + (n_fluid - 1)] =
          specfem::injection::fk_impl::acoustic_from_solid(
              states.elastic_top_storage[f * n_elastic + 0]);
      for (int layer = n_fluid - 2; layer >= 0; --layer) {
        states.acoustic_bottom_storage[f * n_fluid + layer] =
            specfem::injection::fk_impl::acoustic_state_in_layer(
                fluid[layer + 1], omega, fluid[layer + 1].thickness, p,
                states.acoustic_bottom_storage[f * n_fluid + (layer + 1)]);
      }
    }
  }

  return states;
}
