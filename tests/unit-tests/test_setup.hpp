#pragma once

#include "specfem_setup.hpp"

// Compile-time conditional for Google Test expectations
#define EXPECT_REAL_EQ(expected, actual)                                       \
  if constexpr (sizeof(type_real) == sizeof(double)) {                         \
    EXPECT_DOUBLE_EQ(expected, actual);                                        \
  } else {                                                                     \
    EXPECT_FLOAT_EQ(expected, actual);                                         \
  }
