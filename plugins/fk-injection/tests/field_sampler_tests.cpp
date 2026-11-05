#include "field_sampler.hpp"
#include "fk_injection/utilities/cubic_bspline.hpp"
#include "impl/series_layout.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers live in a named namespace (unity-build safe, no anonymous namespace).
// ---------------------------------------------------------------------------
namespace field_sampler_tests_impl {

using fk_injection::type_real;

constexpr int number_of_series = fk_injection::impl::number_of_series;

// Row-major (point, component, step) index into the output of sample().
inline int out_index(int point, int component, int step, int number_of_steps) {
  return (point * fk_injection::number_of_sample_components + component) *
             number_of_steps +
         step;
}

// A table of rows at depths 0, 1, 2, ... m holding the coefficients.
inline std::shared_ptr<const fk_injection::SeriesTable>
make_table(std::vector<type_real> coefficients, int number_of_rows, int ncoef) {
  std::vector<fk_injection::LayerLocation> rows(number_of_rows);
  for (int row = 0; row < number_of_rows; ++row)
    rows[row].depth = row;
  return std::make_shared<const fk_injection::SeriesTable>(
      std::move(rows), ncoef, std::move(coefficients), 0.0);
}

// Output buffer of sample() for a sampler and a window.
inline std::vector<type_real>
make_output(const fk_injection::FieldSampler &result, int number_of_steps) {
  return std::vector<type_real>(result.number_of_points() *
                                fk_injection::number_of_sample_components *
                                number_of_steps);
}

// Writes the spline coefficients of a sampled signal into one series slot.
inline void set_series(std::vector<type_real> &coefficients, int row,
                       int series, int ncoef,
                       const std::vector<double> &signal) {
  std::vector<double> coef(ncoef);
  fk_injection::cubic_bspline::compute_coefficients(signal.data(), ncoef,
                                                    coef.data());
  for (int k = 0; k < ncoef; ++k) {
    coefficients[(row * number_of_series + series) * ncoef + k] =
        static_cast<type_real>(coef[k]);
  }
}

// S0 is the ramp signal[k] = k; S1..S4 are constants 2, 3, 5, 7, each shifted
// by row_shift so that two rows carry different values.
inline void set_row_series(std::vector<type_real> &coefficients, int row,
                           int ncoef, double row_shift) {
  std::vector<double> ramp(ncoef);
  for (int k = 0; k < ncoef; ++k) {
    ramp[k] = static_cast<double>(k);
  }
  set_series(coefficients, row, 0, ncoef, ramp);

  const double constants[4] = { 2.0, 3.0, 5.0, 7.0 };
  for (int series = 1; series < number_of_series; ++series) {
    set_series(coefficients, row, series, ncoef,
               std::vector<double>(ncoef, constants[series - 1] + row_shift));
  }
}

// An elastic point of row 0 with xi1=2 xim=0.5 bulk=0.25 and normal (0,0,1).
inline fk_injection::PointSample make_elastic_point(double offset) {
  fk_injection::PointSample point;
  point.offset = offset;
  point.xi1 = static_cast<type_real>(2.0);
  point.xim = static_cast<type_real>(0.5);
  point.bulk = static_cast<type_real>(0.25);
  point.normal_z = static_cast<type_real>(1.0);
  return point;
}

// 1 row; 2 elastic points with offsets 10.0 and 10.5; cos_phi=0.6
// sin_phi=0.8.
inline fk_injection::FieldSampler
make_synthetic_result(int ncoef = 64, int resampling_rate = 1) {
  std::vector<type_real> coefficients(number_of_series * ncoef);
  set_row_series(coefficients, 0, ncoef, 0.0);

  return fk_injection::FieldSampler(
      make_table(std::move(coefficients), 1, ncoef),
      { make_elastic_point(10.0), make_elastic_point(10.5) },
      static_cast<type_real>(0.6), static_cast<type_real>(0.8),
      resampling_rate);
}

// The what() of the std::runtime_error thrown by the callable; empty if none.
template <typename Callable> std::string rejection_reason(Callable &&callable) {
  try {
    callable();
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "";
}

} // namespace field_sampler_tests_impl

// ---------------------------------------------------------------------------
// Test suite: FieldSampler
// ---------------------------------------------------------------------------

// 1. SampleOffsetAndAzimuth
//    comp0 = cos_phi * S0(a), comp1 = sin_phi * S0(a), comp2 = S1(a) = 2.0
//    where a = k/1 + offset(point).  Checked for interior steps k in [2, 17].
TEST(FieldSampler, SampleOffsetAndAzimuth) {
  const int ncoef = 64;
  const int N = 20;
  const auto result = field_sampler_tests_impl::make_synthetic_result(ncoef, 1);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(0, N, out);

  const double cos_phi = 0.6;
  const double sin_phi = 0.8;
  const double offsets[2] = { 10.0, 10.5 };

  // Avoid spline boundary: k in [2, N-3]; a = k + offset stays in [12, 27.5],
  // well within [2, ncoef-3]=[2,61].
  for (int p = 0; p < 2; ++p) {
    for (int k = 2; k <= N - 3; ++k) {
      const double a = static_cast<double>(k) + offsets[p];
      const double c0 = static_cast<double>(
          out[field_sampler_tests_impl::out_index(p, 0, k, N)]);
      const double c1 = static_cast<double>(
          out[field_sampler_tests_impl::out_index(p, 1, k, N)]);
      const double c2 = static_cast<double>(
          out[field_sampler_tests_impl::out_index(p, 2, k, N)]);
      // comp0 = cos_phi * S0(a) ≈ cos_phi * a  (spline reproduces the ramp)
      EXPECT_NEAR(c0, cos_phi * a, 1.0e-3 * a)
          << "comp0 failed at point=" << p << " k=" << k;
      // comp1 = sin_phi * S0(a)
      EXPECT_NEAR(c1, sin_phi * a, 1.0e-3 * a)
          << "comp1 failed at point=" << p << " k=" << k;
      // comp2 = S1(a) = constant 2.0
      EXPECT_NEAR(c2, 2.0, 1.0e-3)
          << "comp2 failed at point=" << p << " k=" << k;
    }
  }
}

// 2. SampleTractionWeights
//    With S2=3, S3=5, S4=7, xi1=2, xim=0.5, bulk=0.25, n=(0,0,1):
//    sigma_rr = 2*3 - 4*0.5*7 = -8
//    sigma_rz = 5
//    sigma_zz = 3
//    sigma_tt = 0.25*(-8+3) = -1.25
//    With n=(0,0,1): tx=sigma_rz*cos_phi=3.0, ty=sigma_rz*sin_phi=4.0, tz=3.0
//    comp3=tx, comp4=ty, comp5=tz; comp6=0 (elastic)
TEST(FieldSampler, SampleTractionWeights) {
  const int ncoef = 64;
  const int N = 20;
  const auto result = field_sampler_tests_impl::make_synthetic_result(ncoef, 1);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(0, N, out);

  // Constants in S2..S4 → same value at any interior step.
  const int k_check = 10;
  for (int p = 0; p < 2; ++p) {
    EXPECT_NEAR(static_cast<double>(
                    out[field_sampler_tests_impl::out_index(p, 3, k_check, N)]),
                3.0, 1.0e-3)
        << "comp3 (tx) failed at point=" << p;
    EXPECT_NEAR(static_cast<double>(
                    out[field_sampler_tests_impl::out_index(p, 4, k_check, N)]),
                4.0, 1.0e-3)
        << "comp4 (ty) failed at point=" << p;
    EXPECT_NEAR(static_cast<double>(
                    out[field_sampler_tests_impl::out_index(p, 5, k_check, N)]),
                3.0, 1.0e-3)
        << "comp5 (tz) failed at point=" << p;
    // comp6: potential rate = 0 for elastic points
    EXPECT_NEAR(static_cast<double>(
                    out[field_sampler_tests_impl::out_index(p, 6, k_check, N)]),
                0.0, 1.0e-6)
        << "comp6 (potential rate) should be 0 for elastic point=" << p;
  }
}

// 3. SampleNpMapping
//    With resampling_rate=2, the abscissa for SEM step k is k/2 + offset.
//    Verify comp0 = cos_phi * (k/2 + offset) for interior k.
TEST(FieldSampler, SampleNpMapping) {
  const int ncoef = 64;
  const int N = 20;
  // resampling_rate = 2; offsets are 10.0 and 10.5
  const auto result = field_sampler_tests_impl::make_synthetic_result(ncoef, 2);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(0, N, out);

  const double cos_phi = 0.6;
  const double offsets[2] = { 10.0, 10.5 };

  // a = k/2 + offset; for k in [2, N-3] a ranges from [11.0, 20.0], safely
  // within [2, 61].
  for (int p = 0; p < 2; ++p) {
    for (int k = 2; k <= N - 3; ++k) {
      const double a = static_cast<double>(k) / 2.0 + offsets[p];
      const double c0 = static_cast<double>(
          out[field_sampler_tests_impl::out_index(p, 0, k, N)]);
      EXPECT_NEAR(c0, cos_phi * a, 1.0e-3 * a)
          << "NP mapping failed at point=" << p << " k=" << k;
    }
  }
}

// 4. SampleFirstStepShiftsAbscissa
//    A window starting at first_step reads a = (first_step + k) + offset.
TEST(FieldSampler, SampleFirstStepShiftsAbscissa) {
  const int N = 5;
  const int first_step = 7;
  const auto result = field_sampler_tests_impl::make_synthetic_result(64, 1);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(first_step, N, out);

  const double offsets[2] = { 10.0, 10.5 };
  for (int p = 0; p < 2; ++p) {
    for (int k = 0; k < N; ++k) {
      const double a = static_cast<double>(first_step + k) + offsets[p];
      EXPECT_NEAR(static_cast<double>(
                      out[field_sampler_tests_impl::out_index(p, 0, k, N)]),
                  0.6 * a, 1.0e-3 * a)
          << "failed at point=" << p << " k=" << k;
    }
  }
}

// 5. SampleReadsEachPointsRow
//    Row 1 holds S1 = 2 + 100, so a slip in the row stride reads row 0's
//    (or a neighbouring series') constant instead.
TEST(FieldSampler, SampleReadsEachPointsRow) {
  const int ncoef = 64;
  const int N = 3;

  std::vector<fk_injection::type_real> coefficients(
      2 * field_sampler_tests_impl::number_of_series * ncoef);
  field_sampler_tests_impl::set_row_series(coefficients, 0, ncoef, 0.0);
  field_sampler_tests_impl::set_row_series(coefficients, 1, ncoef, 100.0);

  fk_injection::PointSample in_row_0 =
      field_sampler_tests_impl::make_elastic_point(10.0);
  fk_injection::PointSample in_row_1 =
      field_sampler_tests_impl::make_elastic_point(10.0);
  in_row_1.row = 1;

  const fk_injection::FieldSampler result(
      field_sampler_tests_impl::make_table(coefficients, 2, ncoef),
      { in_row_0, in_row_1 }, static_cast<fk_injection::type_real>(0.6),
      static_cast<fk_injection::type_real>(0.8), 1);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(0, N, out);

  EXPECT_NEAR(
      static_cast<double>(out[field_sampler_tests_impl::out_index(0, 2, 1, N)]),
      2.0, 1.0e-3);
  EXPECT_NEAR(
      static_cast<double>(out[field_sampler_tests_impl::out_index(1, 2, 1, N)]),
      102.0, 1.0e-2);
}

// 6. SampleAcousticPoint
//    An acoustic point has zero traction and potential rate = S2 = 3; the
//    kinematic components are recovered as on an elastic point.
TEST(FieldSampler, SampleAcousticPoint) {
  const int ncoef = 64;
  const int N = 3;

  std::vector<fk_injection::type_real> coefficients(
      field_sampler_tests_impl::number_of_series * ncoef);
  field_sampler_tests_impl::set_row_series(coefficients, 0, ncoef, 0.0);

  fk_injection::PointSample acoustic;
  acoustic.offset = 10.0;
  acoustic.medium = specfem::plugin::Medium::acoustic;

  const fk_injection::FieldSampler result(
      field_sampler_tests_impl::make_table(coefficients, 1, ncoef),
      { acoustic }, static_cast<fk_injection::type_real>(0.6),
      static_cast<fk_injection::type_real>(0.8), 1);

  std::vector<fk_injection::type_real> out =
      field_sampler_tests_impl::make_output(result, N);
  result.sample(0, N, out);

  EXPECT_NEAR(static_cast<double>(out[field_sampler_tests_impl::out_index(
                  0, fk_injection::sample_kinematic_offset + 2, 1, N)]),
              2.0, 1.0e-3);
  for (int component = 0; component < 3; ++component) {
    EXPECT_EQ(out[field_sampler_tests_impl::out_index(
                  0, fk_injection::sample_traction_offset + component, 1, N)],
              static_cast<fk_injection::type_real>(0)) // written as an exact 0
        << "traction component " << component;
  }
  EXPECT_NEAR(static_cast<double>(out[field_sampler_tests_impl::out_index(
                  0, fk_injection::sample_potential_rate_offset, 1, N)]),
              3.0, 1.0e-3);
}

// 7. Rejects: wrong output size, inconsistent shape, out-of-range row,
//    unsupported medium, resampling rate below 1.
TEST(FieldSampler, SampleRejectsWrongOutputSize) {
  const auto result = field_sampler_tests_impl::make_synthetic_result();
  std::vector<fk_injection::type_real> out(
      2 * fk_injection::number_of_sample_components * 3 - 1);

  const std::string reason = field_sampler_tests_impl::rejection_reason(
      [&] { result.sample(0, 3, out); });
  EXPECT_NE(reason.find("out has 41 values"), std::string::npos) << reason;
}

TEST(FieldSampler, RejectsCoefficientSizeMismatch) {
  const std::string reason = field_sampler_tests_impl::rejection_reason([] {
    field_sampler_tests_impl::make_table(
        std::vector<fk_injection::type_real>(7), 1, 2);
  });
  EXPECT_NE(reason.find("expected 10"), std::string::npos) << reason;
}

TEST(FieldSampler, RejectsUnsortedRows) {
  const std::string reason = field_sampler_tests_impl::rejection_reason([] {
    fk_injection::SeriesTable(
        std::vector<fk_injection::LayerLocation>{ { .depth = 1000 },
                                                  { .depth = 0 } },
        0, {}, 0.0);
  });
  EXPECT_NE(reason.find("row 1 is out of order"), std::string::npos) << reason;
}

TEST(FieldSampler, RejectsRowOutOfRange) {
  fk_injection::PointSample point;
  point.row = 1;

  const std::string reason = field_sampler_tests_impl::rejection_reason([&] {
    fk_injection::FieldSampler(
        field_sampler_tests_impl::make_table(
            std::vector<fk_injection::type_real>(10), 1, 2),
        { point }, 1, 0, 1);
  });
  EXPECT_NE(reason.find("point 0 has row 1"), std::string::npos) << reason;
}

TEST(FieldSampler, RejectsElasticSpin) {
  fk_injection::PointSample point;
  point.medium = specfem::plugin::Medium::elastic_spin;

  const std::string reason = field_sampler_tests_impl::rejection_reason([&] {
    fk_injection::FieldSampler(
        field_sampler_tests_impl::make_table(
            std::vector<fk_injection::type_real>(10), 1, 2),
        { point }, 1, 0, 1);
  });
  EXPECT_NE(reason.find("acoustic or elastic"), std::string::npos) << reason;
}

TEST(FieldSampler, RejectsResamplingRateBelowOne) {
  const std::string reason = field_sampler_tests_impl::rejection_reason([] {
    fk_injection::FieldSampler(
        field_sampler_tests_impl::make_table(
            std::vector<fk_injection::type_real>(10), 1, 2),
        {}, 1, 0, 0);
  });
  EXPECT_NE(reason.find("resampling_rate"), std::string::npos) << reason;
}
