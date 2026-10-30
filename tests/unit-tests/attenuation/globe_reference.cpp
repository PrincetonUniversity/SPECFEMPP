#include "globe_model_evaluator.h"
#include "specfem/attenuation.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

TEST(GlobeAttenuationReference, IndependentQFitAndModulusFactors) {
  using specfem::units::unit_symbols::Hz;
  constexpr int n = specfem::constants::N_SLS;
  const specfem::utilities::Band<specfem::units::Hertz> band(0.001 * Hz,
                                                             0.1 * Hz);
  const auto sigma = specfem::attenuation::compute_tau_sigma<n>(band);
  for (const double q : { -1., 0., 65., 80., 184., 600., 9000., 9999.,
                          std::numeric_limits<double>::infinity() }) {
    double reference_sigma[n], reference_epsilon[n], reference_rates[n];
    double relaxed_scale, ratio;
    globe_evaluator_attenuation_reference(
        band.min.raw(), band.max.raw(), q, 0.01, 1.0, reference_sigma,
        reference_epsilon, &relaxed_scale, &ratio, reference_rates);
    const auto epsilon =
        specfem::attenuation::globe::compute_tau_eps<n>(q, sigma, band);
    const auto factors =
        specfem::attenuation::globe::get_attenuation_property_values<n>(
            sigma, epsilon);
    const auto scales =
        specfem::attenuation::globe::get_attenuation_modulus_scales<n>(
            0.01, epsilon, sigma, q, 1.0);
    const auto scale = scales.unrelaxed();
    const double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 1.e-4;
    SCOPED_TRACE(q);
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(sigma(j), reference_sigma[j], tolerance * reference_sigma[j]);
      EXPECT_NEAR(epsilon(j), reference_epsilon[j],
                  tolerance * reference_epsilon[j]);
      EXPECT_NEAR(2 * factors.beta(j) / sigma(j), reference_rates[j],
                  tolerance * std::abs(reference_rates[j]) +
                      4 * std::numeric_limits<type_real>::epsilon() / sigma(j));
    }
    EXPECT_NEAR(scales.relaxed, relaxed_scale, tolerance * relaxed_scale);
    EXPECT_NEAR(scales.ratio, ratio, tolerance * ratio);
    EXPECT_NEAR(factors.one_minus_sum_beta, ratio, tolerance * ratio);
    EXPECT_NEAR(scale, relaxed_scale * ratio,
                tolerance * relaxed_scale * ratio);
  }
}
