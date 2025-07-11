#pragma once
#include <gtest/gtest.h>

// For better naming
struct Serial : std::integral_constant<bool, false> {};
struct SIMD : std::integral_constant<bool, true> {};

using TestTypes = ::testing::Types<Serial, SIMD>;
