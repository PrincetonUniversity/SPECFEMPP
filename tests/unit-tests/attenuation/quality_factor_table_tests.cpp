#include "specfem/attenuation/quality_factor_table.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace quality_factor_table_test_impl {
// A cheap stand-in for the attenuation fit: records the Q it was given.
struct Fit {
  type_real q;
};
struct Counter {
  int *calls;
  Fit operator()(const type_real q) const {
    ++*calls;
    return { q };
  }
};
} // namespace quality_factor_table_test_impl

TEST(Attenuation_QualityFactorTable, BucketsAtResolution) {
  specfem::attenuation::QualityFactorTable<int> table;
  EXPECT_EQ(table.bucket(150.31), 1503);
  EXPECT_EQ(table.bucket(0.05), 1);
  // Without a maximum, Q is used as given.
  EXPECT_NEAR(table.clip(12345), 12345, 0.0);
  EXPECT_EQ(table.bucket(12345), 123450);
  EXPECT_THROW(table.clip(std::numeric_limits<type_real>::quiet_NaN()),
               std::invalid_argument);
}

TEST(Attenuation_QualityFactorTable, FirstQRepresentsItsBucket) {
  specfem::attenuation::QualityFactorTable<int> table;
  EXPECT_NEAR(table.representative(100.01), static_cast<type_real>(100.01),
              0.0);
  EXPECT_NEAR(table.representative(100.09), static_cast<type_real>(100.01),
              0.0);
  EXPECT_NEAR(table.representative(100.14), static_cast<type_real>(100.14),
              0.0);
}

TEST(Attenuation_QualityFactorTable, MaximumClipsBeforeBucketing) {
  specfem::attenuation::QualityFactorTable<int> table(9000);
  const type_real inf = std::numeric_limits<type_real>::infinity();
  for (const type_real q :
       { type_real(0), type_real(-1), type_real(9999), inf }) {
    EXPECT_NEAR(table.clip(q), 9000, 0.0);
    EXPECT_EQ(table.bucket(q), 90000);
    EXPECT_NEAR(table.representative(q), 9000, 0.0);
  }
  EXPECT_EQ(table.bucket(9000), 90000);
}

TEST(Attenuation_QualityFactorTable, FitsOncePerBucket) {
  using quality_factor_table_test_impl::Counter;
  using quality_factor_table_test_impl::Fit;
  int calls = 0;
  const Counter fit{ &calls };
  specfem::attenuation::QualityFactorTable<Fit> table;

  // Same bucket: one fit, of the first Q.
  const Fit &a = table.fit(100.01, fit);
  const Fit &b = table.fit(100.09, fit);
  EXPECT_EQ(&a, &b);
  EXPECT_NEAR(b.q, static_cast<type_real>(100.01), 0.0);
  EXPECT_EQ(calls, 1);

  // Adjacent bucket: a new fit.
  const Fit &c = table.fit(100.14, fit);
  EXPECT_NEAR(c.q, static_cast<type_real>(100.14), 0.0);
  EXPECT_EQ(calls, 2);

  // representative() alone does not fit; fit() then uses that Q.
  EXPECT_NEAR(table.representative(200.05), static_cast<type_real>(200.05),
              0.0);
  EXPECT_EQ(calls, 2);
  EXPECT_NEAR(table.fit(200.0, fit).q, static_cast<type_real>(200.05), 0.0);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(table.fits(), calls);
}

TEST(Attenuation_QualityFactorTable, FitCountIndependentOfPointCount) {
  using quality_factor_table_test_impl::Counter;
  using quality_factor_table_test_impl::Fit;
  int calls = 0;
  specfem::attenuation::QualityFactorTable<Fit> table;
  // 100000 "GLL points" with Q in [80, 90): at most 100 buckets.
  for (int i = 0; i < 100000; ++i) {
    table.fit(80 + (i % 1000) * 0.01, Counter{ &calls });
  }
  EXPECT_EQ(calls, 100);
  EXPECT_EQ(table.fits(), 100);
}
