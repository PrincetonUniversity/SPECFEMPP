#pragma once
#include "specfem_setup.hpp"
#include <string>

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
