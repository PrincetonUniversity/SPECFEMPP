#include "specfem/attenuation.hpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace globe_formulation_test_impl {
template <int N>
using View = Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>;

template <int N> View<N> make(const std::array<double, N> &values) {
  View<N> view("values");
  for (int i = 0; i < N; ++i)
    view(i) = values[i];
  return view;
}
} // namespace globe_formulation_test_impl

TEST(GlobeAttenuation_PropertyValues, RatioIsOnePlusSumOfDefects) {
  namespace test = globe_formulation_test_impl;
  const auto sigma = test::make<3>({ 1.0, 2.0, 5.0 });
  const auto epsilon = test::make<3>({ 2.0, 3.0, 10.0 });
  const auto globe =
      specfem::attenuation::globe::get_attenuation_property_values<3>(sigma,
                                                                      epsilon);
  const auto legacy =
      specfem::attenuation::get_attenuation_property_values<3>(sigma, epsilon);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(globe.beta(i), legacy.beta(i)));
  }
  // Globe: 1 + (1 + 0.5 + 1). The 2D routine keeps sum(tau_eps/tau_sigma).
  EXPECT_TRUE(specfem::utilities::is_close(globe.one_minus_sum_beta,
                                           static_cast<type_real>(3.5)));
  EXPECT_TRUE(specfem::utilities::is_close(legacy.one_minus_sum_beta,
                                           static_cast<type_real>(5.5)));
}

TEST(GlobeAttenuation_PropertyValues, NoAttenuationHasUnitRatio) {
  namespace test = globe_formulation_test_impl;
  const auto tau = test::make<3>({ 1.0, 2.0, 5.0 });
  const auto values =
      specfem::attenuation::globe::get_attenuation_property_values<3>(tau, tau);
  // Globe evaluates 1 - tau_eps * (1/tau_sigma), so round-off remains.
  const double tolerance = 4 * std::numeric_limits<type_real>::epsilon();
  EXPECT_NEAR(values.one_minus_sum_beta, 1, 3 * tolerance);
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(values.beta(i), 0, tolerance);
}

TEST(GlobeAttenuation_ModulusScales, UniformDefectAtUnitOmegaTau) {
  namespace test = globe_formulation_test_impl;
  constexpr double two_pi = 2.0 * std::numbers::pi;
  const auto sigma = test::make<3>({ 1 / two_pi, 1 / two_pi, 1 / two_pi });
  const auto epsilon =
      test::make<3>({ 4 / (3 * two_pi), 4 / (3 * two_pi), 4 / (3 * two_pi) });
  const auto scales =
      specfem::attenuation::globe::get_attenuation_modulus_scales<3>(
          1.0, epsilon, sigma, 100.0, 1.0);
  // At omega=1: A=13/25, B=9/25; relaxed-to-unrelaxed ratio 2.
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-12 : 1.e-5;
  EXPECT_NEAR(scales.ratio, 2.0, tolerance);
  EXPECT_NEAR(scales.relaxed * scales.ratio, (13.0 + std::sqrt(250.0)) / 25.0,
              tolerance);
}

TEST(GlobeAttenuation_ModulusScales, TwoMechanismsWithLogCorrection) {
  namespace test = globe_formulation_test_impl;
  const auto sigma = test::make<2>({ 1.0, 2.0 });
  const auto epsilon = test::make<2>({ 1.5, 2.5 });
  const double fc = 1.0 / (2.0 * std::numbers::pi);
  const auto scales =
      specfem::attenuation::globe::get_attenuation_modulus_scales<2>(
          fc, epsilon, sigma, 100.0, fc);
  // A=225/377, B=84/377; ratio 7/4; no log correction since fc == f0.
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-12 : 1.e-5;
  EXPECT_NEAR(scales.ratio, 1.75, tolerance);
  EXPECT_NEAR(scales.relaxed, (225.0 + std::sqrt(57681.0)) / 754.0, tolerance);
}

TEST(GlobeAttenuation_ModulusScales, RangeCheckAppliesToRelaxedScale) {
  namespace test = globe_formulation_test_impl;
  const auto tau = test::make<3>({ 1.0, 1.0, 1.0 });
  // factor_scale_mu0 = 1 + 2 ln(10) / (2 pi) > 1.5
  EXPECT_THROW(specfem::attenuation::globe::get_attenuation_modulus_scales<3>(
                   10.0, tau, tau, 2.0, 1.0),
               std::runtime_error);
  // A nonpositive reference frequency disables the log correction.
  const auto scales =
      specfem::attenuation::globe::get_attenuation_modulus_scales<3>(
          10.0, tau, tau, 2.0, 0.0);
  EXPECT_NEAR(scales.relaxed, 1.0, 0.0);
}

TEST(GlobeAttenuation_Maxwell, UnitRelaxedModulus) {
  Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<double[3], Kokkos::LayoutRight, Kokkos::HostSpace> sigma(
      "sigma"),
      epsilon("epsilon");
  f(0) = 1.0 / (2.0 * std::numbers::pi);
  Kokkos::deep_copy(sigma, 1.0);
  Kokkos::deep_copy(epsilon, 2.0);
  const auto moduli =
      specfem::attenuation::globe::maxwell<1, 3>(f, sigma, epsilon);
  // Each mechanism contributes 3/2 and 1/2 at omega=1; A = 1 - 3 + 3*(3/2).
  EXPECT_NEAR(moduli.real(0), 2.5, 1.e-14);
  EXPECT_NEAR(moduli.imag(0), 1.5, 1.e-14);
}

TEST(GlobeAttenuation_Objective, PenalizesZeroStorageModulus) {
  specfem::attenuation::globe::AttenuationObjective<1> objective;
  objective.iQ = 0.01;
  objective.f = decltype(objective.f)("f");
  objective.tau_sigma = decltype(objective.tau_sigma)("sigma");
  Kokkos::deep_copy(objective.f, 1.0 / (2.0 * std::numbers::pi));
  Kokkos::deep_copy(objective.tau_sigma, 1.0);
  Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace> epsilon(
      "epsilon");
  epsilon(0) = -1.0;
  const double penalty = objective(epsilon);
  EXPECT_TRUE(std::isfinite(penalty));
  EXPECT_GT(penalty, 1.e30);
}

TEST(GlobeAttenuation_ComputeTauEps, FitsTargetQ) {
  using specfem::units::unit_symbols::Hz;
  const specfem::utilities::Band<specfem::units::Hertz> band(0.01 * Hz,
                                                             1.0 * Hz);
  const auto sigma = specfem::attenuation::compute_tau_sigma<3>(band);
  const auto epsilon =
      specfem::attenuation::globe::compute_tau_eps<3>(100.0, sigma, band);
  Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<double[3], Kokkos::LayoutRight, Kokkos::HostSpace> s("s"),
      e("e");
  for (int i = 0; i < 3; ++i) {
    s(i) = sigma(i);
    e(i) = epsilon(i);
    EXPECT_GT(epsilon(i), sigma(i));
  }
  f(0) = 0.1; // band center
  const auto moduli = specfem::attenuation::globe::maxwell<1, 3>(f, s, e);
  // Three mechanisms over two decades approximate constant Q within ~10%.
  EXPECT_NEAR(moduli.real(0) / moduli.imag(0), 100.0, 10.0);
}

TEST(GlobeAttenuation_ComputeTauEps, RejectsUnconvergedFit) {
  using specfem::units::unit_symbols::Hz;
  const specfem::utilities::Band<specfem::units::Hertz> band(0.01 * Hz,
                                                             1.0 * Hz);
  const auto sigma = specfem::attenuation::compute_tau_sigma<2>(band);
  // Two mechanisms cannot finish this strong-attenuation fit within globe's
  // 400-evaluation budget; globe aborts.
  EXPECT_THROW(
      specfem::attenuation::globe::compute_tau_eps<2>(1.0, sigma, band),
      std::runtime_error);
}

TEST(GlobeAttenuation_QualityFactor, ClipsToMaximum) {
  namespace globe = specfem::attenuation::globe;
  const type_real inf = std::numeric_limits<type_real>::infinity();
  EXPECT_NEAR(globe::clip_quality_factor(80), 80, 0.0);
  EXPECT_NEAR(globe::clip_quality_factor(9000), 9000, 0.0);
  EXPECT_NEAR(globe::clip_quality_factor(0), 9000, 0.0);
  EXPECT_NEAR(globe::clip_quality_factor(-5), 9000, 0.0);
  EXPECT_NEAR(globe::clip_quality_factor(9999), 9000, 0.0);
  EXPECT_NEAR(globe::clip_quality_factor(inf), 9000, 0.0);
  EXPECT_THROW(
      globe::clip_quality_factor(std::numeric_limits<type_real>::quiet_NaN()),
      std::invalid_argument);

  // The globe table clips exactly as the fit does.
  auto table = globe::make_quality_factor_table<2>();
  for (const type_real q :
       { type_real(80), type_real(0), type_real(-5), type_real(9999), inf }) {
    EXPECT_NEAR(table.clip(q), globe::clip_quality_factor(q), 0.0);
  }
  EXPECT_EQ(table.bucket(inf), 90000);
}

namespace globe_formulation_test_impl {
/** @brief Globe fit of one Q in a global-scale band, as the solver runs it. */
struct GlobeFit {
  specfem::utilities::Band<specfem::units::Hertz> band;
  View<3> tau_sigma;
  type_real fc;

  GlobeFit()
      : band((1.0 / 15632.376) * specfem::units::unit_symbols::Hz,
             (1.0 / 277.987305) * specfem::units::unit_symbols::Hz),
        tau_sigma(specfem::attenuation::compute_tau_sigma<3>(band)),
        fc(std::sqrt(band.min.raw() * band.max.raw())) {}

  specfem::attenuation::globe::QualityFactorFit<3>
  operator()(const type_real q) const {
    return specfem::attenuation::globe::fit_quality_factor<3>(q, tau_sigma,
                                                              band, fc, 1.0);
  }
};
} // namespace globe_formulation_test_impl

TEST(GlobeAttenuation_QualityFactorTables, OneFitPerBucket) {
  const globe_formulation_test_impl::GlobeFit fit;
  specfem::attenuation::globe::QualityFactorTables<3> tables;

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

TEST(GlobeAttenuation_QualityFactorTables, BucketErrorIsBoundedByStep) {
  // A point whose Q is up to one bucket width (0.1) from its representative
  // gets the representative's fit. The modulus defect scales like 1/Q, so
  // the error is about 0.1/Q relative: below 1e-3 at Q = 100.
  const globe_formulation_test_impl::GlobeFit fit;
  const auto representative = fit(100.01);
  const auto exact = fit(100.09);
  for (int j = 0; j < 3; ++j) {
    const double defect_r = representative.tau_eps(j) / fit.tau_sigma(j) - 1.0;
    const double defect_e = exact.tau_eps(j) / fit.tau_sigma(j) - 1.0;
    EXPECT_NEAR(defect_r, defect_e, 1.e-3 * std::abs(defect_e));
  }
  // The dispersion shift 1 - scale also scales like 1/Q.
  const double shift = 1.0 - exact.scales.unrelaxed();
  EXPECT_NEAR(representative.scales.unrelaxed(), exact.scales.unrelaxed(),
              1.e-3 * std::abs(shift));
}
