#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/halfspace_solve.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

// ============================================================================
// Helper: build a standard elastic half-space + one finite layer setup.
//
//   hs   : rho=3300, vp=8100, vs=4500 (half-space)
//   layer: rho=2800, vp=6000, vs=3500, H=10000 m
//   p    : sin(20 deg) / 8100 s/m
//   omega: 2*pi*0.5 rad/s, incident P wave
// ============================================================================
namespace field_recovery_tests_impl {

inline void
make_elastic_model(specfem::injection::fk::ElasticIsotropicLayer &hs,
                   specfem::injection::fk::ElasticIsotropicLayer &layer,
                   type_real &p, Kokkos::complex<double> &omega,
                   specfem::utilities::ComplexMatrix<4> &e_mat,
                   specfem::utilities::ComplexVector<4> &modes) {
  hs.density = static_cast<type_real>(3300.0);
  hs.p_velocity = static_cast<type_real>(8100.0);
  hs.s_velocity = static_cast<type_real>(4500.0);
  hs.thickness = static_cast<type_real>(0.0);

  layer.density = static_cast<type_real>(2800.0);
  layer.p_velocity = static_cast<type_real>(6000.0);
  layer.s_velocity = static_cast<type_real>(3500.0);
  layer.thickness = static_cast<type_real>(10000.0);

  constexpr double deg20_rad = 20.0 * 3.141592653589793 / 180.0;
  p = static_cast<type_real>(Kokkos::sin(deg20_rad) / 8100.0);

  omega = Kokkos::complex<double>(2.0 * 3.141592653589793 * 0.5, 0.0);

  e_mat = specfem::injection::fk_impl::elastic_halfspace_eigenmatrix(hs, p);

  const specfem::injection::fk::incident_wave_type wave_type =
      specfem::injection::fk::incident_wave_type::p;
  const auto amp = specfem::injection::fk_impl::incident_amplitude(
      hs, wave_type, static_cast<type_real>(1.0), p);

  specfem::utilities::ComplexMatrix<4> qmat_zero;
  const auto coeffs = specfem::injection::fk_impl::halfspace_coefficients(
      e_mat, false, qmat_zero, wave_type, amp.c1, amp.c3);

  modes = specfem::injection::fk_impl::build_bottom_vector(wave_type, amp.c1,
                                                           amp.c3, coeffs);
}

} // namespace field_recovery_tests_impl

// ============================================================================
// Test 5: HalfspaceZeroDepthEqualsModesProjection
//
// At zz=0, G(0)=I so elastic_state_in_halfspace == E * modes component-wise.
// ============================================================================

TEST(FieldRecovery, HalfspaceZeroDepthEqualsModesProjection) {
  specfem::injection::fk::ElasticIsotropicLayer hs;
  specfem::injection::fk::ElasticIsotropicLayer layer;
  type_real p;
  Kokkos::complex<double> omega;
  specfem::utilities::ComplexMatrix<4> e_mat;
  specfem::utilities::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  const auto state = specfem::injection::fk_impl::elastic_state_in_halfspace(
      e_mat, hs, omega, p, /*zz=*/0.0, modes);

  const auto expected = e_mat * modes;

  constexpr double kTol = 1.0e-8;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(state[i].real(), expected[i].real(), kTol)
        << "component " << i << " real";
    EXPECT_NEAR(state[i].imag(), expected[i].imag(), kTol)
        << "component " << i << " imag";
  }
}

// ============================================================================
// Test 6: ElasticInterfaceContinuity
//
// elastic_state_in_layer at height=0 returns state_at_bottom unchanged,
// because gamma0 * Pbar(0) = I.
// ============================================================================

TEST(FieldRecovery, ElasticInterfaceContinuity) {
  specfem::injection::fk::ElasticIsotropicLayer hs;
  specfem::injection::fk::ElasticIsotropicLayer layer;
  type_real p;
  Kokkos::complex<double> omega;
  specfem::utilities::ComplexMatrix<4> e_mat;
  specfem::utilities::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  specfem::utilities::ComplexVector<4> state_at_bottom;
  state_at_bottom[0] = Kokkos::complex<double>(1.0, 0.5);
  state_at_bottom[1] = Kokkos::complex<double>(-0.3, 0.2);
  state_at_bottom[2] = Kokkos::complex<double>(0.1, -0.4);
  state_at_bottom[3] = Kokkos::complex<double>(0.7, 0.0);

  const auto result = specfem::injection::fk_impl::elastic_state_in_layer(
      layer, omega, /*height=*/static_cast<type_real>(0.0), p, state_at_bottom);

  constexpr double kTol = 1.0e-9;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(result[i].real(), state_at_bottom[i].real(), kTol)
        << "component " << i << " real";
    EXPECT_NEAR(result[i].imag(), state_at_bottom[i].imag(), kTol)
        << "component " << i << " imag";
  }
}

// ============================================================================
// Test 9: AcousticFreeSurfacePressureSmall
//
// Sanity check: project_acoustic at height=500 m produces a finite pressure.
// ============================================================================

TEST(FieldRecovery, AcousticFreeSurfacePressureSmall) {
  specfem::injection::fk::ElasticIsotropicLayer hs;
  specfem::injection::fk::ElasticIsotropicLayer layer;
  type_real p;
  Kokkos::complex<double> omega;
  specfem::utilities::ComplexMatrix<4> e_mat;
  specfem::utilities::ComplexVector<4> modes;
  field_recovery_tests_impl::make_elastic_model(hs, layer, p, omega, e_mat,
                                                modes);

  specfem::injection::fk::AcousticLayer fluid;
  fluid.density = static_cast<type_real>(1020.0);
  fluid.p_velocity = static_cast<type_real>(1500.0);
  fluid.thickness = static_cast<type_real>(2000.0);

  const auto solid_interface_state = e_mat * modes;
  const specfem::injection::fk_impl::AcousticState base =
      specfem::injection::fk_impl::acoustic_from_solid(solid_interface_state);

  const specfem::injection::fk_impl::AcousticState at =
      specfem::injection::fk_impl::acoustic_state_in_layer(
          fluid, omega, static_cast<type_real>(500.0), p, base);

  const specfem::injection::fk_impl::SeriesSpectrum s =
      specfem::injection::fk_impl::project_acoustic(
          at, static_cast<double>(fluid.density), omega, p);

  const Kokkos::complex<double> pres =
      s.values[specfem::injection::fk_impl::series_sigma_zz];
  EXPECT_TRUE(std::isfinite(pres.real())) << "pressure real part is not finite";
  EXPECT_TRUE(std::isfinite(pres.imag())) << "pressure imag part is not finite";
}
