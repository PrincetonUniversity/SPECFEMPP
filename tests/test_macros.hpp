#pragma once
#include "specfem/setup.hpp"
#include <string>

// Workaround for GTest EXPECT_NO_THROW duplicate label error with GCC 14
// This assumes gtest.h is included before this file or will be included.
// If included AFTER, this might not work unless we ensure order.
// Best to include this file in test files after gtest.
#define LOCAL_EXPECT_NO_THROW(stmt)                                            \
  try {                                                                        \
    stmt;                                                                      \
  } catch (...) {                                                              \
    ADD_FAILURE() << "Expected: " #stmt                                        \
                     " doesn't throw. Throws unknown exception.";              \
  }

#define LOCAL_EXPECT_THROW(stmt, etype)                                        \
  try {                                                                        \
    stmt;                                                                      \
    ADD_FAILURE() << "Expected exception " #etype " not thrown";               \
  } catch (const etype &) {                                                    \
  } catch (...) {                                                              \
    ADD_FAILURE() << "Expected exception " #etype " but threw something else"; \
  }

// Compile-time conditional for Google Test expectations
#define EXPECT_REAL_EQ(expected, actual)                                       \
  if constexpr (sizeof(type_real) == sizeof(double)) {                         \
    EXPECT_DOUBLE_EQ(expected, actual);                                        \
  } else {                                                                     \
    EXPECT_FLOAT_EQ(expected, actual);                                         \
  }

// Helper struct for expected vs got comparison output
template <typename T, typename U> struct ExpectedGot {
  const T &expected;
  const U &got;
  ExpectedGot(const T &exp, const U &g) : expected(exp), got(g) {}
};

// Helper function to create ExpectedGot
template <typename T, typename U>
ExpectedGot<T, U> expected_got(const T &expected, const U &got) {
  return ExpectedGot<T, U>(expected, got);
}

// Operator<< for ExpectedGot in global namespace for ADL
template <typename T, typename U>
std::ostream &operator<<(std::ostream &os, const ExpectedGot<T, U> &eg) {
  return os << "Expected: \n"
            << eg.expected << "\nGot:      " << eg.got << "\n";
}
