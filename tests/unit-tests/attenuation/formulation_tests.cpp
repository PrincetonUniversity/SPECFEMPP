#include "specfem/attenuation.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

// The attenuation formulation against its defining relations: a 1/L
// generalized Maxwell solid with flat Q(omega) over the band, the storage
// modulus at the band center matched to the absorption band, M_U/M_R as the
// high-frequency limit, and relaxation rates from the relaxed modulus.

namespace formulation_test_impl {
template <int N>
using View = Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>;
template <int N>
using DoubleView =
    Kokkos::View<double[N], Kokkos::LayoutRight, Kokkos::HostSpace>;

/** @brief Fit of one Q in a band, as the solver runs it. */
template <int N_SLS> struct Fit {
  specfem::utilities::Band<specfem::units::Hertz> band;
  View<N_SLS> tau_sigma;
  type_real fc;
  type_real f0;

  Fit(const double min_frequency, const double max_frequency,
      const double reference_frequency)
      : band(min_frequency * specfem::units::unit_symbols::Hz,
             max_frequency * specfem::units::unit_symbols::Hz),
        tau_sigma(specfem::attenuation::compute_tau_sigma<N_SLS>(band)),
        fc(std::sqrt(min_frequency * max_frequency)), f0(reference_frequency) {}

  specfem::attenuation::QualityFactorFit<N_SLS>
  operator()(const type_real q) const {
    return specfem::attenuation::fit_quality_factor<N_SLS>(q, tau_sigma, band,
                                                           fc, f0);
  }

  /** @brief Storage and loss moduli of a fit, M_R = 1, at one frequency. */
  specfem::attenuation::MaxwellFactors<1, double>
  moduli(const specfem::attenuation::QualityFactorFit<N_SLS> &fit,
         const double frequency) const {
    DoubleView<1> f("f");
    DoubleView<N_SLS> sigma("sigma"), epsilon("epsilon");
    f(0) = frequency;
    for (int i = 0; i < N_SLS; ++i) {
      sigma(i) = tau_sigma(i);
      epsilon(i) = fit.tau_eps(i);
    }
    return specfem::attenuation::maxwell<1, N_SLS, double>(f, sigma, epsilon);
  }
};

/** @brief A global-scale band (periods 278 s to 15632 s), f0 = 1 Hz. */
inline Fit<3> globe_band() {
  return { 1.0 / 15632.376, 1.0 / 277.987305, 1.0 };
}

/** @brief A local-scale band (0.1 Hz to 5.6 Hz) with f0 at its center. */
inline Fit<3> cartesian_band() {
  return { 0.1, 0.1 * std::pow(10.0, 1.75), 0.1 * std::pow(10.0, 0.875) };
}
} // namespace formulation_test_impl

class Attenuation_Formulation : public ::testing::TestWithParam<double> {};

// Q(omega) = A/B is flat over the band to within the error of three
// mechanisms over 1.75 decades.
TEST_P(Attenuation_Formulation, QualityFactorIsFlatOverBand) {
  const double q = GetParam();
  for (const auto &band : { formulation_test_impl::globe_band(),
                            formulation_test_impl::cartesian_band() }) {
    const auto fit = band(q);
    const double fmin = band.band.min.raw();
    const double fmax = band.band.max.raw();
    for (int i = 0; i < 21; ++i) {
      const double f = fmin * std::pow(fmax / fmin, i / 20.0);
      const auto moduli = band.moduli(fit, f);
      EXPECT_NEAR(moduli.real(0) / moduli.imag(0), q, 0.05 * q) << "f = " << f;
    }
  }
}

// M_1(omega_c) = M_R A(omega_c) is the absorption-band modulus
// M_0 [1 + 2 ln(f_c / f_0) / (pi Q)].
TEST_P(Attenuation_Formulation, StorageModulusMatchesAbsorptionBandAtCenter) {
  const double q = GetParam();
  for (const auto &band : { formulation_test_impl::globe_band(),
                            formulation_test_impl::cartesian_band() }) {
    const auto fit = band(q);
    const auto moduli = band.moduli(fit, band.fc);
    const double expected =
        1.0 + 2.0 * std::log(static_cast<double>(band.fc) / band.f0) /
                  (std::numbers::pi * q);
    EXPECT_NEAR(fit.scales.relaxed * moduli.real(0), expected, 1e-5 * expected);
  }
}

// M_U/M_R = (1/L) sum tau_eps/tau_sigma is the high-frequency limit of A and
// is stored the same way in the scales and the relaxation coefficients.
TEST_P(Attenuation_Formulation, UnrelaxedRatioIsHighFrequencyLimit) {
  const double q = GetParam();
  const auto band = formulation_test_impl::globe_band();
  const auto fit = band(q);
  double ratio = 0;
  for (int i = 0; i < 3; ++i)
    ratio += static_cast<double>(fit.tau_eps(i)) / band.tau_sigma(i) / 3;
  EXPECT_NEAR(fit.scales.ratio, ratio, 1e-6 * ratio);
  EXPECT_NEAR(fit.properties.unrelaxed_ratio, ratio, 1e-6 * ratio);
  EXPECT_NEAR(band.moduli(fit, 1e8).real(0), ratio, 1e-6 * ratio);
  EXPECT_GT(ratio, 1.0);
  EXPECT_NEAR(fit.scales.unrelaxed(), fit.scales.relaxed * fit.scales.ratio,
              1e-6);
}

// Memory variable j relaxes at M_R (tau_eps_j / tau_sigma_j - 1) / L /
// tau_sigma_j, with M_R = M_U / (M_U/M_R).
TEST_P(Attenuation_Formulation, RelaxationRatesUseRelaxedModulus) {
  const double q = GetParam();
  const auto band = formulation_test_impl::cartesian_band();
  const auto fit = band(q);
  const double unrelaxed = 3.0e10;
  const double relaxed = unrelaxed / fit.scales.ratio;
  for (int j = 0; j < 3; ++j) {
    const double sigma = band.tau_sigma(j);
    const double expected =
        relaxed * (fit.tau_eps(j) / sigma - 1.0) / 3.0 / sigma;
    EXPECT_NEAR(fit.properties.relaxation_rate(unrelaxed, j, sigma), expected,
                1e-5 * std::abs(expected));
  }
}

INSTANTIATE_TEST_SUITE_P(Q, Attenuation_Formulation,
                         ::testing::Values(20.0, 80.0, 312.0, 9000.0));

TEST(Attenuation_FormulationFit, IdentityLeavesModulusUnattenuated) {
  const auto fit = specfem::attenuation::QualityFactorFit<3>::identity();
  EXPECT_NEAR(fit.scales.unrelaxed(), 1.0, 0.0);
  EXPECT_NEAR(fit.properties.unrelaxed_ratio, 1.0, 0.0);
  for (int j = 0; j < 3; ++j)
    EXPECT_NEAR(fit.properties.relaxation_rate(1.0, j, 1.0), 0.0, 0.0);
}

TEST(Attenuation_FormulationFit, ObjectivePenalizesZeroStorageModulus) {
  specfem::attenuation::AttenuationObjective<1> objective;
  objective.iQ = 0.01;
  objective.f = decltype(objective.f)("f");
  objective.tau_sigma = decltype(objective.tau_sigma)("sigma");
  Kokkos::deep_copy(objective.f, 1.0 / (2.0 * std::numbers::pi));
  Kokkos::deep_copy(objective.tau_sigma, 1.0);
  formulation_test_impl::DoubleView<1> epsilon("epsilon");
  // A = (1 + tau_eps) / 2 at omega * tau_sigma = 1.
  epsilon(0) = -1.0;
  const double penalty = objective(epsilon);
  EXPECT_TRUE(std::isfinite(penalty));
  EXPECT_GT(penalty, 1.e30);
}

TEST(Attenuation_FormulationFit, RejectsUnconvergedFit) {
  using specfem::units::unit_symbols::Hz;
  const specfem::utilities::Band<specfem::units::Hertz> band(0.01 * Hz,
                                                             1.0 * Hz);
  const auto sigma = specfem::attenuation::compute_tau_sigma<2>(band);
  // Two mechanisms cannot finish this strong-attenuation fit within the
  // 400-evaluation budget.
  EXPECT_THROW(specfem::attenuation::compute_tau_eps<2>(1.0, sigma, band),
               std::runtime_error);
}

TEST(Attenuation_QualityFactorFitTables, ClipsToMaximumByDefault) {
  const type_real inf = std::numeric_limits<type_real>::infinity();
  specfem::attenuation::QualityFactorFitTables<3> clipped;
  const std::pair<type_real, type_real> cases[] = {
    { 80, 80 },   { 9000, 9000 }, { 0, 9000 },
    { -5, 9000 }, { 9999, 9000 }, { inf, 9000 }
  };
  for (const auto &[q, expected] : cases) {
    EXPECT_NEAR(clipped.bulk.clip(q), expected, 0.0);
    EXPECT_NEAR(clipped.shear.clip(q), expected, 0.0);
  }
  EXPECT_EQ(clipped.shear.bucket(inf), 90000);
  EXPECT_THROW(clipped.shear.clip(std::numeric_limits<type_real>::quiet_NaN()),
               std::invalid_argument);

  // A nonpositive maximum (2D) uses Q as given.
  specfem::attenuation::QualityFactorFitTables<3> unclipped(0);
  EXPECT_NEAR(unclipped.shear.clip(9999), 9999, 0.0);
}

TEST(Attenuation_QualityFactorFitTables, OneFitPerBucket) {
  const auto fit = formulation_test_impl::globe_band();
  specfem::attenuation::QualityFactorFitTables<3> tables;

  // 100.01 and 100.09 share a bucket: the second gets the first's fit.
  const auto &first = tables.shear.fit(100.01, fit);
  const auto &same = tables.shear.fit(100.09, fit);
  EXPECT_EQ(&first, &same);
  EXPECT_EQ(tables.fits(), 1);

  // 100.04 and 100.14 are in adjacent buckets: two different fits.
  const auto &lower = tables.bulk.fit(100.04, fit);
  const auto &upper = tables.bulk.fit(100.14, fit);
  EXPECT_EQ(tables.fits(), 3);
  bool differs = false;
  for (int j = 0; j < 3; ++j)
    differs = differs || lower.tau_eps(j) != upper.tau_eps(j);
  EXPECT_TRUE(differs);

  // Q outside (0, 9000] is clipped into the last bucket, which is fitted at
  // Q = 9000 once.
  for (const type_real q :
       { type_real(0), type_real(-5), type_real(9000), type_real(9999) }) {
    EXPECT_EQ(tables.shear.bucket(q), 90000);
    EXPECT_NEAR(tables.shear.representative(q), 9000, 0.0);
    tables.shear.fit(q, fit);
  }
  EXPECT_EQ(tables.fits(), 4);
}

TEST(Attenuation_QualityFactorFitTables, BucketErrorIsBoundedByStep) {
  // A point whose Q is up to one bucket width (0.1) from its representative
  // gets the representative's fit. The modulus defect scales like 1/Q, so
  // the error is about 0.1/Q relative: below 1e-3 at Q = 100.
  const auto fit = formulation_test_impl::globe_band();
  const auto representative = fit(100.01);
  const auto exact = fit(100.09);
  for (int j = 0; j < 3; ++j) {
    const double defect_r = representative.properties.beta(j);
    const double defect_e = exact.properties.beta(j);
    EXPECT_NEAR(defect_r, defect_e, 1.e-3 * std::abs(defect_e));
  }
  // The dispersion shift 1 - scale also scales like 1/Q.
  const double shift = 1.0 - exact.scales.unrelaxed();
  EXPECT_NEAR(representative.scales.unrelaxed(), exact.scales.unrelaxed(),
              1.e-3 * std::abs(shift));
}
