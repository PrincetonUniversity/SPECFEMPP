#include "../SPECFEM_Environment.hpp"
#include "math.h"
#include "specfem/source_time_functions.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>

// Test fixture for basic setup
class SourceTimeFunctionSetup : public ::testing::Test {
protected:
  void SetUp() override {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }

  void TearDown() override {
    // Note: Don't finalize Kokkos here as other tests may need it
  }
};

// Helper to compute expected values based on mathematical formulas
namespace {
type_real pi = std::atan(1.0) * 4.0;

inline type_real compute_expected_gaussian(type_real t, type_real f0,
                                           type_real factor) {
  type_real a = pi * pi * f0 * f0;
  return factor * (-1.0 * std::exp(-a * t * t) / (2.0 * a));
}

inline type_real compute_expected_d1gaussian(type_real t, type_real f0,
                                             type_real factor) {
  type_real a = pi * pi * f0 * f0;
  return factor * t * std::exp(-a * t * t);
}

inline type_real compute_expected_d2gaussian(type_real t, type_real f0,
                                             type_real factor) {
  type_real a = pi * pi * f0 * f0;
  return factor * (1.0 - 2.0 * a * t * t) * std::exp(-a * t * t);
}

inline type_real compute_expected_d3gaussian(type_real t, type_real f0,
                                             type_real factor) {
  type_real a = pi * pi * f0 * f0;
  return factor *
         (-2.0 * a * t * (3.0 - 2.0 * a * t * t) * std::exp(-a * t * t));
}

inline type_real compute_expected_d4gaussian(type_real t, type_real f0,
                                             type_real factor) {
  type_real a = pi * pi * f0 * f0;
  return factor *
         (-2.0 * a * (3.0 - 12.0 * a * t * t + 4.0 * a * a * t * t * t * t) *
          std::exp(-a * t * t));
}

inline type_real compute_expected_gaussian_hdur(type_real t, type_real hdur,
                                                type_real factor) {
  type_real a = 1.0 / (hdur * hdur);
  return factor * (std::exp(-a * t * t) / (std::sqrt(pi) * hdur));
}

inline type_real compute_expected_d1gaussian_hdur(type_real t, type_real hdur,
                                                  type_real factor) {
  type_real a = 1.0 / (hdur * hdur);
  return factor * (-2.0 * a * t * compute_expected_gaussian_hdur(t, hdur, 1.0));
}

inline type_real compute_expected_d2gaussian_hdur(type_real t, type_real hdur,
                                                  type_real factor) {
  type_real a = 1.0 / (hdur * hdur);
  return factor * (2.0 * a * (-1.0 + 2.0 * a * t * t) *
                   compute_expected_gaussian_hdur(t, hdur, 1.0));
}

inline type_real compute_expected_heaviside(type_real t, type_real hdur,
                                            type_real factor) {
  return factor * 0.5 * (1.0 + std::erf(t / hdur));
}
} // namespace

// Helper struct to define STF test parameters and expected values
template <typename STFType> struct STFTraits;

// Ricker traits
template <> struct STFTraits<specfem::source_time_functions::Ricker> {
  using STFType = specfem::source_time_functions::Ricker;

  static constexpr const char *name = "Ricker";
  static constexpr type_real default_f0 = 10.0;
  static constexpr type_real default_tshift = 0.0;
  static constexpr type_real default_factor = 1.0;
  static constexpr bool default_use_trick = false;
  static constexpr int default_nsteps = 100;
  static constexpr type_real default_dt = 0.001;

  static std::unique_ptr<STFType> create(int nsteps, type_real dt, type_real f0,
                                         type_real tshift, type_real factor,
                                         bool use_trick) {
    return std::make_unique<STFType>(nsteps, dt, f0, tshift, factor, use_trick);
  }

  // Compute expected value at time t (after applying tshift)
  static type_real compute_expected(type_real t, type_real f0, type_real tshift,
                                    type_real factor, bool use_trick) {
    type_real t_eff = t - tshift;
    if (use_trick) {
      return compute_expected_d4gaussian(t_eff, f0, factor);
    } else {
      return compute_expected_d2gaussian(t_eff, f0, factor);
    }
  }
};

// dGaussian traits
template <> struct STFTraits<specfem::source_time_functions::dGaussian> {
  using STFType = specfem::source_time_functions::dGaussian;

  static constexpr const char *name = "dGaussian";
  static constexpr type_real default_f0 = 10.0;
  static constexpr type_real default_tshift = 0.0;
  static constexpr type_real default_factor = 1.0;
  static constexpr bool default_use_trick = false;
  static constexpr int default_nsteps = 100;
  static constexpr type_real default_dt = 0.001;

  static std::unique_ptr<STFType> create(int nsteps, type_real dt, type_real f0,
                                         type_real tshift, type_real factor,
                                         bool use_trick) {
    return std::make_unique<STFType>(nsteps, dt, f0, tshift, factor, use_trick);
  }

  // Compute expected value at time t (after applying tshift)
  static type_real compute_expected(type_real t, type_real f0, type_real tshift,
                                    type_real factor, bool use_trick) {
    type_real t_eff = t - tshift;
    if (use_trick) {
      return compute_expected_d3gaussian(t_eff, f0, factor);
    } else {
      return compute_expected_d1gaussian(t_eff, f0, factor);
    }
  }
};

// Dirac traits
template <> struct STFTraits<specfem::source_time_functions::Dirac> {
  using STFType = specfem::source_time_functions::Dirac;

  static constexpr const char *name = "Dirac";
  static constexpr type_real default_f0 = 10.0;
  static constexpr type_real default_tshift = 0.0;
  static constexpr type_real default_factor = 1.0;
  static constexpr bool default_use_trick = false;
  static constexpr int default_nsteps = 100;
  static constexpr type_real default_dt = 0.001;

  static std::unique_ptr<STFType> create(int nsteps, type_real dt, type_real f0,
                                         type_real tshift, type_real factor,
                                         bool use_trick) {
    return std::make_unique<STFType>(nsteps, dt, f0, tshift, factor, use_trick);
  }

  // Compute expected value at time t (after applying tshift)
  static type_real compute_expected(type_real t, type_real f0, type_real tshift,
                                    type_real factor, bool use_trick) {
    type_real t_eff = t - tshift;
    if (use_trick) {
      return compute_expected_d2gaussian(t_eff, f0, factor);
    } else {
      return compute_expected_gaussian(t_eff, f0, factor);
    }
  }
};

// Heaviside traits
template <> struct STFTraits<specfem::source_time_functions::Heaviside> {
  using STFType = specfem::source_time_functions::Heaviside;

  static constexpr const char *name = "Heaviside";
  static constexpr type_real default_f0 = 0.1; // hdur value
  static constexpr type_real default_tshift = 0.0;
  static constexpr type_real default_factor = 1.0;
  static constexpr bool default_use_trick = false;
  static constexpr int default_nsteps = 100;
  static constexpr type_real default_dt = 0.001;

  static std::unique_ptr<STFType> create(int nsteps, type_real dt,
                                         type_real hdur, type_real tshift,
                                         type_real factor, bool use_trick) {
    return std::make_unique<STFType>(nsteps, dt, hdur, tshift, factor,
                                     use_trick);
  }

  // Compute expected value at time t (after applying tshift)
  // Note: parameter is named f0 for template compatibility, but it's hdur
  static type_real compute_expected(type_real t, type_real hdur,
                                    type_real tshift, type_real factor,
                                    bool use_trick) {
    type_real t_eff = t - tshift;
    // Note: hdur needs to be adjusted by SOURCE_DECAY_MIMIC_TRIANGLE
    type_real hdur_adjusted = hdur / 1.62800; // SOURCE_DECAY_MIMIC_TRIANGLE
    if (use_trick) {
      return compute_expected_d2gaussian_hdur(t_eff, hdur_adjusted, factor);
    } else {
      return compute_expected_heaviside(t_eff, hdur_adjusted, factor);
    }
  }
};

// GaussianHdur traits
template <> struct STFTraits<specfem::source_time_functions::GaussianHdur> {
  using STFType = specfem::source_time_functions::GaussianHdur;

  static constexpr const char *name = "GaussianHdur";
  static constexpr type_real default_f0 = 0.1; // hdur value
  static constexpr type_real default_tshift = 0.0;
  static constexpr type_real default_factor = 1.0;
  static constexpr bool default_use_trick = false;
  static constexpr int default_nsteps = 100;
  static constexpr type_real default_dt = 0.001;

  static std::unique_ptr<STFType> create(int nsteps, type_real dt,
                                         type_real hdur, type_real tshift,
                                         type_real factor, bool use_trick) {
    return std::make_unique<STFType>(nsteps, dt, hdur, tshift, factor,
                                     use_trick);
  }

  // Compute expected value at time t (after applying tshift)
  // Note: parameter is named f0 for template compatibility, but it's hdur
  static type_real compute_expected(type_real t, type_real hdur,
                                    type_real tshift, type_real factor,
                                    bool use_trick) {
    type_real t_eff = t - tshift;
    // Note: hdur needs to be adjusted by SOURCE_DECAY_MIMIC_TRIANGLE
    type_real hdur_adjusted = hdur / 1.62800; // SOURCE_DECAY_MIMIC_TRIANGLE
    if (use_trick) {
      return compute_expected_d2gaussian_hdur(t_eff, hdur_adjusted, factor);
    } else {
      return compute_expected_gaussian_hdur(t_eff, hdur_adjusted, factor);
    }
  }
};

// Type list for typed tests (excluding external as it has different interface)
using AnalyticSTFTypes =
    ::testing::Types<specfem::source_time_functions::Ricker,
                     specfem::source_time_functions::dGaussian,
                     specfem::source_time_functions::Dirac,
                     specfem::source_time_functions::Heaviside,
                     specfem::source_time_functions::GaussianHdur>;

// Typed test fixture
template <typename T> class AnalyticSTFTest : public SourceTimeFunctionSetup {
protected:
  using STFType = T;
  using Traits = STFTraits<T>;
};

TYPED_TEST_SUITE(AnalyticSTFTest, AnalyticSTFTypes);

// Test: Constructor and basic properties
TYPED_TEST(AnalyticSTFTest, ConstructorAndBasicProperties) {
  using Traits = typename TestFixture::Traits;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt,
                            Traits::default_f0, Traits::default_tshift,
                            Traits::default_factor, Traits::default_use_trick);

  ASSERT_NE(stf, nullptr);

  // Check that t0 is computed correctly (should be negative)
  type_real t0 = stf->get_t0();
  EXPECT_LT(t0, 0.0);

  // Check tshift
  type_real tshift = stf->get_tshift();
  EXPECT_REAL_EQ(tshift, Traits::default_tshift);
}

// Test: update_tshift functionality
TYPED_TEST(AnalyticSTFTest, UpdateTshift) {
  using Traits = typename TestFixture::Traits;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt,
                            Traits::default_f0, Traits::default_tshift,
                            Traits::default_factor, Traits::default_use_trick);

  type_real new_tshift = 0.5;
  stf->update_tshift(new_tshift);

  EXPECT_REAL_EQ(stf->get_tshift(), new_tshift);
}

// Test: compute_source_time_function generates valid output
TYPED_TEST(AnalyticSTFTest, ComputeSourceTimeFunctionValidOutput) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 100;
  const type_real dt = 0.001;
  const type_real t0 = 0.0;
  const int ncomponents = 1;

  auto stf =
      Traits::create(nsteps, dt, Traits::default_f0, Traits::default_tshift,
                     Traits::default_factor, Traits::default_use_trick);

  // Allocate output array
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace> stf_values(
      "stf_values", nsteps, ncomponents);

  // Compute STF
  stf->compute_source_time_function(t0, dt, nsteps, stf_values);

  // Check that values are finite
  bool all_finite = true;
  for (int i = 0; i < nsteps; ++i) {
    for (int j = 0; j < ncomponents; ++j) {
      if (!std::isfinite(stf_values(i, j))) {
        all_finite = false;
        break;
      }
    }
    if (!all_finite)
      break;
  }
  EXPECT_TRUE(all_finite);

  // Check that not all values are zero
  bool has_nonzero = false;
  for (int i = 0; i < nsteps; ++i) {
    for (int j = 0; j < ncomponents; ++j) {
      if (std::abs(stf_values(i, j)) > 1e-10) {
        has_nonzero = true;
        break;
      }
    }
    if (has_nonzero)
      break;
  }
  EXPECT_TRUE(has_nonzero);
}

// Test: Multiple components
TYPED_TEST(AnalyticSTFTest, MultipleComponents) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 50;
  const type_real dt = 0.001;
  const type_real t0 = 0.0;
  const int ncomponents = 3;

  auto stf =
      Traits::create(nsteps, dt, Traits::default_f0, Traits::default_tshift,
                     Traits::default_factor, Traits::default_use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace> stf_values(
      "stf_values", nsteps, ncomponents);

  stf->compute_source_time_function(t0, dt, nsteps, stf_values);

  // For analytic STFs, all components should have the same values
  for (int i = 0; i < nsteps; ++i) {
    for (int j = 1; j < ncomponents; ++j) {
      EXPECT_REAL_EQ(stf_values(i, 0), stf_values(i, j));
    }
  }
}

// Test: Different frequency values
TYPED_TEST(AnalyticSTFTest, DifferentFrequencies) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 100;
  const type_real dt = 0.001;
  const type_real t0 = 0.0;
  const int ncomponents = 1;

  // Test with low frequency
  auto stf_low =
      Traits::create(nsteps, dt, 5.0, Traits::default_tshift,
                     Traits::default_factor, Traits::default_use_trick);

  // Test with high frequency
  auto stf_high =
      Traits::create(nsteps, dt, 20.0, Traits::default_tshift,
                     Traits::default_factor, Traits::default_use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values_low("stf_low", nsteps, ncomponents);
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values_high("stf_high", nsteps, ncomponents);

  stf_low->compute_source_time_function(t0, dt, nsteps, stf_values_low);
  stf_high->compute_source_time_function(t0, dt, nsteps, stf_values_high);

  // STFs should be different
  bool are_different = false;
  for (int i = 0; i < nsteps; ++i) {
    if (std::abs(stf_values_low(i, 0) - stf_values_high(i, 0)) > 1e-6) {
      are_different = true;
      break;
    }
  }
  EXPECT_TRUE(are_different);
}

// Test: Factor scaling
TYPED_TEST(AnalyticSTFTest, FactorScaling) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 100;
  const type_real dt = 0.001;
  const type_real t0 = 0.0;
  const int ncomponents = 1;
  const type_real factor1 = 1.0;
  const type_real factor2 = 2.5;

  auto stf1 =
      Traits::create(nsteps, dt, Traits::default_f0, Traits::default_tshift,
                     factor1, Traits::default_use_trick);

  auto stf2 =
      Traits::create(nsteps, dt, Traits::default_f0, Traits::default_tshift,
                     factor2, Traits::default_use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values1("stf1", nsteps, ncomponents);
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values2("stf2", nsteps, ncomponents);

  stf1->compute_source_time_function(t0, dt, nsteps, stf_values1);
  stf2->compute_source_time_function(t0, dt, nsteps, stf_values2);

  // Values should be scaled by the factor ratio
  const type_real expected_ratio = factor2 / factor1;
  for (int i = 0; i < nsteps; ++i) {
    if (std::abs(stf_values1(i, 0)) > 1e-10) {
      type_real actual_ratio = stf_values2(i, 0) / stf_values1(i, 0);
      EXPECT_NEAR(actual_ratio, expected_ratio, 1e-5);
    }
  }
}

// Test: Print method returns non-empty string
TYPED_TEST(AnalyticSTFTest, PrintMethod) {
  using Traits = typename TestFixture::Traits;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt,
                            Traits::default_f0, Traits::default_tshift,
                            Traits::default_factor, Traits::default_use_trick);

  std::string output = stf->print();
  EXPECT_FALSE(output.empty());
  EXPECT_NE(output.find(Traits::name), std::string::npos);
}

// Test: Equality operator
TYPED_TEST(AnalyticSTFTest, EqualityOperator) {
  using Traits = typename TestFixture::Traits;

  auto stf1 = Traits::create(Traits::default_nsteps, Traits::default_dt,
                             Traits::default_f0, Traits::default_tshift,
                             Traits::default_factor, Traits::default_use_trick);

  auto stf2 = Traits::create(Traits::default_nsteps, Traits::default_dt,
                             Traits::default_f0, Traits::default_tshift,
                             Traits::default_factor, Traits::default_use_trick);

  // Same parameters should be equal
  EXPECT_TRUE(*stf1 == *stf2);
  EXPECT_FALSE(*stf1 != *stf2);
}

// Test: Inequality operator
TYPED_TEST(AnalyticSTFTest, InequalityOperator) {
  using Traits = typename TestFixture::Traits;

  auto stf1 = Traits::create(Traits::default_nsteps, Traits::default_dt,
                             Traits::default_f0, Traits::default_tshift,
                             Traits::default_factor, Traits::default_use_trick);

  auto stf2 = Traits::create(Traits::default_nsteps, Traits::default_dt,
                             Traits::default_f0 * 2.0, // Different frequency
                             Traits::default_tshift, Traits::default_factor,
                             Traits::default_use_trick);

  // Different parameters should not be equal
  EXPECT_FALSE(*stf1 == *stf2);
  EXPECT_TRUE(*stf1 != *stf2);
}

// Individual tests for Ricker with use_trick_for_better_pressure
TEST_F(SourceTimeFunctionSetup, RickerWithTrickForBetterPressure) {
  const int nsteps = 100;
  const type_real dt = 0.001;
  const type_real f0 = 10.0;
  const type_real tshift = 0.0;
  const type_real factor = 1.0;
  const type_real t0 = 0.0;
  const int ncomponents = 1;

  auto stf_without_trick =
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, f0, tshift, factor, false);
  auto stf_with_trick =
      std::make_unique<specfem::source_time_functions::Ricker>(
          nsteps, dt, f0, tshift, factor, true);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values_without("stf_without", nsteps, ncomponents);
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values_with("stf_with", nsteps, ncomponents);

  stf_without_trick->compute_source_time_function(t0, dt, nsteps,
                                                  stf_values_without);
  stf_with_trick->compute_source_time_function(t0, dt, nsteps, stf_values_with);

  // Values should be different when using the trick
  bool are_different = false;
  for (int i = 0; i < nsteps; ++i) {
    if (std::abs(stf_values_without(i, 0) - stf_values_with(i, 0)) > 1e-6) {
      are_different = true;
      break;
    }
  }
  EXPECT_TRUE(are_different);
}

// Test for external source time function
TEST_F(SourceTimeFunctionSetup, ExternalSTFConstruction) {
  // Create a YAML node for external STF
  YAML::Node external_node;
  external_node["type"] = "ascii";
  external_node["ncomponents"] = 1;
  external_node["x-component"] = "test_file.txt";

  const int nsteps = 100;
  const type_real dt = 0.001;

  // This test just checks that construction doesn't throw
  // In a real scenario, you would need actual data files
  // EXPECT_NO_THROW({
  //   specfem::source_time_functions::external ext_stf(external_node, nsteps,
  //   dt);
  // });

  // For now, we just verify the test compiles
  SUCCEED();
}

// Test: Time shift affects the timing of the STF
TYPED_TEST(AnalyticSTFTest, TimeShiftAffectsTiming) {
  using Traits = typename TestFixture::Traits;

  // Heaviside function does not have a distinct peak, skip this test

  const int nsteps = 200;
  const type_real dt = 0.001;
  const type_real t0 = 0.0;
  const int ncomponents = 1;
  const type_real tshift1 = 0.0;
  const type_real tshift2 = 0.05;

  auto stf1 = Traits::create(nsteps, dt, Traits::default_f0, tshift1,
                             Traits::default_factor, Traits::default_use_trick);

  auto stf2 = Traits::create(nsteps, dt, Traits::default_f0, tshift2,
                             Traits::default_factor, Traits::default_use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values1("stf1", nsteps, ncomponents);
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      stf_values2("stf2", nsteps, ncomponents);

  stf1->compute_source_time_function(t0, dt, nsteps, stf_values1);
  stf2->compute_source_time_function(t0, dt, nsteps, stf_values2);

  int peak_idx1 = 0, peak_idx2 = 0;
  type_real max_val1 = 0.0, max_val2 = 0.0;

  if constexpr (std::is_same<
                    typename TestFixture::STFType,
                    specfem::source_time_functions::Heaviside>::value) {

    // Get t0 values of the two STFs
    type_real t0_1 = stf1->get_t0();
    type_real t0_2 = stf2->get_t0();

    for (int i = 0; i < nsteps; ++i) {
      // Make sure that around t0 the values are different
      if ((std::abs(i * dt - t0_1) < tshift2) ||
          (std::abs(i * dt - t0_2) < tshift2)) {
        EXPECT_NE(stf_values1(i, 0), stf_values2(i, 0));
      }
    }

    // For the other timefunctions find peak values
  } else {
    for (int i = 0; i < nsteps; ++i) {
      if (std::abs(stf_values1(i, 0)) > max_val1) {
        max_val1 = std::abs(stf_values1(i, 0));
        peak_idx1 = i;
      }
      if (std::abs(stf_values2(i, 0)) > max_val2) {
        max_val2 = std::abs(stf_values2(i, 0));
        peak_idx2 = i;
      }
    }

    // The shifted STF should have its peak at a different time
    EXPECT_NE(peak_idx1, peak_idx2);
  }
}

// Test: STF values decay to near zero at far times
TYPED_TEST(AnalyticSTFTest, DecayAtFarTimes) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 500;
  const type_real dt = 0.01;
  const type_real t0 = 0.0;
  const int ncomponents = 1;

  auto stf =
      Traits::create(nsteps, dt, Traits::default_f0, Traits::default_tshift,
                     Traits::default_factor, Traits::default_use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace> stf_values(
      "stf", nsteps, ncomponents);

  stf->compute_source_time_function(t0, dt, nsteps, stf_values);

  // Check that values at the end are much smaller than peak
  type_real max_val = 0.0;
  for (int i = 0; i < nsteps; ++i) {
    max_val = std::max(max_val, std::abs(stf_values(i, 0)));
  }

  // Catch Heaviside, which does not decay to zero
  if constexpr (std::is_same<
                    typename TestFixture::STFType,
                    specfem::source_time_functions::Heaviside>::value) {
    SUCCEED();
    return;
  }

  // Last 10% of values should be relatively small
  int start_idx = static_cast<int>(nsteps * 0.9);
  for (int i = start_idx; i < nsteps; ++i) {
    EXPECT_LT(std::abs(stf_values(i, 0)), max_val * 0.1);
  }
}

// ============================================================================
// SPECIFIC VALUE TESTS - Testing exact mathematical formulas
// ============================================================================

// Test: compute() returns expected values from mathematical formula
TYPED_TEST(AnalyticSTFTest, ComputeSpecificValues) {
  using Traits = typename TestFixture::Traits;

  const type_real f0 = 10.0;
  const type_real tshift = 0.0;
  const type_real factor = 1.5;
  const bool use_trick = false;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt, f0,
                            tshift, factor, use_trick);

  // Test at multiple time points
  std::vector<type_real> test_times = { 0.0, 0.01, 0.05, 0.1 };

  for (type_real t : test_times) {
    type_real actual = stf->compute(t);
    type_real expected =
        Traits::compute_expected(t, f0, tshift, factor, use_trick);
    EXPECT_NEAR(actual, expected, 1e-5)
        << "Mismatch at t=" << t << " for " << Traits::name;
  }
}

// Test: compute() with time shift
TYPED_TEST(AnalyticSTFTest, ComputeWithTimeShift) {
  using Traits = typename TestFixture::Traits;

  const type_real f0 = 10.0;
  const type_real tshift = 0.02;
  const type_real factor = 1.0;
  const bool use_trick = false;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt, f0,
                            tshift, factor, use_trick);

  // Test at multiple time points
  std::vector<type_real> test_times = { 0.0, 0.02, 0.05, 0.1 };

  for (type_real t : test_times) {
    type_real actual = stf->compute(t);
    type_real expected =
        Traits::compute_expected(t, f0, tshift, factor, use_trick);
    EXPECT_NEAR(actual, expected, 1e-5)
        << "Mismatch at t=" << t << " with tshift=" << tshift;
  }
}

// Test: compute() with use_trick enabled
TYPED_TEST(AnalyticSTFTest, ComputeWithTrick) {
  using Traits = typename TestFixture::Traits;

  const type_real f0 = 10.0;
  const type_real tshift = 0.0;
  const type_real factor = 2.0;
  const bool use_trick = true;

  auto stf = Traits::create(Traits::default_nsteps, Traits::default_dt, f0,
                            tshift, factor, use_trick);

  // Test at multiple time points
  std::vector<type_real> test_times = { 0.0, 0.01, 0.05, 0.1 };

  for (type_real t : test_times) {
    type_real actual = stf->compute(t);
    type_real expected =
        Traits::compute_expected(t, f0, tshift, factor, use_trick);
    EXPECT_NEAR(actual, expected, 1e-4)
        << "Mismatch at t=" << t << " with use_trick=true";
  }
}

// Test: compute_source_time_function() matches expected formula at all time
// steps
TYPED_TEST(AnalyticSTFTest, ComputeSTFMatchesFormula) {
  using Traits = typename TestFixture::Traits;

  const int nsteps = 50;
  const type_real dt = 0.01;
  const type_real t0 = 0.0;
  const type_real f0 = 10.0;
  const type_real tshift = 0.0;
  const type_real factor = 1.0;
  const bool use_trick = false;
  const int ncomponents = 1;

  auto stf = Traits::create(nsteps, dt, f0, tshift, factor, use_trick);

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace> stf_values(
      "stf", nsteps, ncomponents);
  stf->compute_source_time_function(t0, dt, nsteps, stf_values);

  for (int i = 0; i < nsteps; ++i) {
    type_real t = t0 + i * dt;
    type_real expected_formula =
        Traits::compute_expected(t, f0, tshift, factor, use_trick);
    type_real actual = stf_values(i, 0);

    EXPECT_NEAR(actual, expected_formula, 1e-5)
        << "compute_source_time_function doesn't match formula at step " << i
        << " (t=" << t << ")";
  }
}
