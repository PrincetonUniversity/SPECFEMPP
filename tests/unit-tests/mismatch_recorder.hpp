#pragma once

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace specfem_test {

/**
 * @brief Folds the entry-wise comparisons of one loop into one assertion.
 *
 * Every entry is still compared, but only the number of mismatches and the
 * first mismatch are kept, so a fully wrong array reports once instead of
 * once per entry.
 *
 * @code
 * specfem_test::MismatchRecorder mismatches;
 * for (...)
 *   mismatches.near(actual(e, iz, ix), expected(e, iz, ix), 1e-4, e, iz, ix);
 * EXPECT_TRUE(mismatches.result()) << view_name;
 * @endcode
 */
class MismatchRecorder {
public:
  /**
   * @brief Record whether two values agree within an absolute tolerance.
   *
   * @param actual Value under test
   * @param expected Reference value
   * @param tolerance Largest accepted absolute difference
   * @param indices Position of the entry, reported for the first mismatch
   */
  template <typename Actual, typename Expected, typename... Indices>
  void near(const Actual &actual, const Expected &expected,
            const double tolerance, const Indices &...indices) {
    const double difference =
        std::abs(static_cast<double>(actual) - static_cast<double>(expected));
    // Written so that a NaN difference counts as a mismatch
    const bool matches = difference <= tolerance;
    record(matches, actual, expected, indices...);
    if (!matches && number_of_mismatches_ == 1) {
      std::ostringstream message;
      message << ", tolerance " << tolerance;
      first_mismatch_ += message.str();
    }
  }

  /**
   * @brief Record whether two values compare equal.
   *
   * @param actual Value under test
   * @param expected Reference value
   * @param indices Position of the entry, reported for the first mismatch
   */
  template <typename Actual, typename Expected, typename... Indices>
  void equal(const Actual &actual, const Expected &expected,
             const Indices &...indices) {
    record(actual == expected, actual, expected, indices...);
  }

  /**
   * @brief Outcome of all recorded comparisons.
   *
   * @return Success when no entry differed, otherwise a failure naming the
   * number of mismatches and the first one
   */
  ::testing::AssertionResult result() const {
    if (number_of_mismatches_ == 0) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << number_of_mismatches_ << " of " << number_of_entries_
           << " entries differ; " << first_mismatch_;
  }

private:
  template <typename Actual, typename Expected, typename... Indices>
  void record(const bool matches, const Actual &actual,
              const Expected &expected, const Indices &...indices) {
    ++number_of_entries_;
    if (matches) {
      return;
    }
    if (number_of_mismatches_++ > 0) {
      return;
    }
    std::ostringstream message;
    message << "first at (";
    std::size_t position = 0;
    ((message << (position++ == 0 ? "" : ", ") << indices), ...);
    message << "): actual " << actual << ", expected " << expected;
    first_mismatch_ = message.str();
  }

  std::size_t number_of_entries_ = 0;
  std::size_t number_of_mismatches_ = 0;
  std::string first_mismatch_;
};

} // namespace specfem_test
