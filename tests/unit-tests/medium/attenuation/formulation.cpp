#include "specfem/attenuation.hpp"
#include "specfem/medium/dim2/elastic/isotropic/material.hpp"
#include <gtest/gtest.h>
#include <type_traits>

// 2D keeps the 1/L-normalized routines; 3D uses SPECFEM3D_GLOBE's.
TEST(AttenuationFormulation, MaterialsSelectFormulationByDimension) {
  using specfem::units::unit_symbols::Hz;
  constexpr int n = specfem::constants::N_SLS;
  const specfem::utilities::Band<specfem::units::Hertz> band(0.01 * Hz,
                                                             1.0 * Hz);
  const auto sigma = specfem::attenuation::compute_tau_sigma<n>(band);
  constexpr type_real qkappa = 300, qmu = 80, f0 = 1.0, fc = 0.1;

  const specfem::medium_container::impl::ComputedAttenuationValues<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic>
      dim2(qkappa, qmu, f0, fc, band, sigma);
  const specfem::medium_container::impl::ComputedAttenuationValues<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic>
      dim3(qkappa, qmu, f0, fc, band, sigma);

  const auto legacy_epsilon =
      specfem::attenuation::compute_tau_eps<n>(qmu, sigma, band);
  const auto legacy_values =
      specfem::attenuation::get_attenuation_property_values<n>(sigma,
                                                               legacy_epsilon);
  const auto globe_epsilon =
      specfem::attenuation::globe::compute_tau_eps<n>(qmu, sigma, band);
  const auto globe_values =
      specfem::attenuation::globe::get_attenuation_property_values<n>(
          sigma, globe_epsilon);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(dim2.tau_epsilon_mu(i), legacy_epsilon(i), 0.0);
    EXPECT_NEAR(dim3.tau_epsilon_mu(i), globe_epsilon(i), 0.0);
  }
  EXPECT_NEAR(dim2.mu_attenuation_properties.one_minus_sum_beta,
              legacy_values.one_minus_sum_beta, 0.0);
  EXPECT_NEAR(dim3.mu_attenuation_properties.one_minus_sum_beta,
              globe_values.one_minus_sum_beta, 0.0);
  EXPECT_NEAR(dim2.mu_scale,
              specfem::attenuation::get_attenuation_scale_factor<n>(
                  fc, legacy_epsilon, sigma, qmu, f0),
              0.0);
  EXPECT_NEAR(dim3.mu_scale,
              specfem::attenuation::globe::get_attenuation_modulus_scales<n>(
                  fc, globe_epsilon, sigma, qmu, f0)
                  .unrelaxed(),
              0.0);
  // The two formulations really differ: Sigma(tau_eps/tau_sigma) vs 1+Sigma.
  EXPECT_GT(dim2.mu_attenuation_properties.one_minus_sum_beta, 2.5);
  EXPECT_LT(dim3.mu_attenuation_properties.one_minus_sum_beta, 1.5);
}
