// Unit tests for specfem/tag_dispatch/find_in.hpp
//
// Covers:
//   find_in_v<combos, search>  — constexpr index of first matching combo
//   find_in_t<combos, search>  — integral_constant<size_t, I> or void
//
// All correctness proofs are expressed as static_asserts (compile-time),
// backed by matching runtime EXPECT_* assertions for readable failure output.

#include "specfem/tag_dispatch/element_combinations.hpp"
#include "specfem/tag_dispatch/find_in.hpp"
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>

using D = specfem::element::dimension_tag;
using M = specfem::element::medium_tag;
using P = specfem::element::property_tag;
using A = specfem::element::attenuation_tag;
using B = specfem::element::boundary_tag;

namespace td = specfem::tag_dispatch;

// SmallET:
//   combos[0] = (dim2, elastic_psv, isotropic, none, none)
//   combos[1] = (dim2, acoustic,    isotropic, none, none)
using SmallET = td::element_combinations<
    td::dimension_set<D::dim2>, td::medium_set<M::elastic_psv, M::acoustic>,
    td::property_set<P::isotropic>, td::attenuation_set<A::none>,
    td::boundary_set<B::none> >;

// Search values
constexpr auto search_psv = SmallET::combos[0];
constexpr auto search_ac = SmallET::combos[1];

// A combo with valid types but not present in SmallET
// (SmallET only has A::none; A::constant_isotropic won't be found)
constexpr SmallET::combo_type search_missing{ D::dim2, M::elastic_psv,
                                              P::isotropic,
                                              A::constant_isotropic, B::none };

// ── find_in_v: correct index ──────────────────────────────────────────

TEST(FindIn, FirstCombo_Index0) {
  constexpr std::size_t idx = td::find_in_v<SmallET::combos, search_psv>;
  static_assert(idx == 0);
  EXPECT_EQ(idx, std::size_t(0));
}

TEST(FindIn, SecondCombo_Index1) {
  constexpr std::size_t idx = td::find_in_v<SmallET::combos, search_ac>;
  static_assert(idx == 1);
  EXPECT_EQ(idx, std::size_t(1));
}

// ── find_in_t: type result ────────────────────────────────────────────

TEST(FindIn, IntegralConstantType_ForFirstCombo) {
  using Result = td::find_in_t<SmallET::combos, search_psv>;
  static_assert(
      std::is_same_v<Result, std::integral_constant<std::size_t, 0> >);
  EXPECT_TRUE(
      (std::is_same_v<Result, std::integral_constant<std::size_t, 0> >));
}

TEST(FindIn, IntegralConstantType_ForSecondCombo) {
  using Result = td::find_in_t<SmallET::combos, search_ac>;
  static_assert(
      std::is_same_v<Result, std::integral_constant<std::size_t, 1> >);
  EXPECT_TRUE(
      (std::is_same_v<Result, std::integral_constant<std::size_t, 1> >));
}

TEST(FindIn, NotFound_YieldsVoid) {
  using Result = td::find_in_t<SmallET::combos, search_missing>;
  static_assert(std::is_void_v<Result>);
  EXPECT_TRUE((std::is_void_v<Result>));
}

// ── Self-consistency: find_in_v(combos[i]) == i ───────────────────────

TEST(FindIn, SelfConsistent_WithCombos) {
  static_assert(td::find_in_v<SmallET::combos, SmallET::combos[0]> == 0);
  static_assert(td::find_in_v<SmallET::combos, SmallET::combos[1]> == 1);
  EXPECT_EQ((td::find_in_v<SmallET::combos, SmallET::combos[0]>),
            std::size_t(0));
  EXPECT_EQ((td::find_in_v<SmallET::combos, SmallET::combos[1]>),
            std::size_t(1));
}

// ── Larger combo array ────────────────────────────────────────────────

using LargerET = td::element_combinations<
    td::dimension_set<D::dim2, D::dim3>,
    td::medium_set<M::elastic_psv, M::acoustic, M::elastic> >;
// combos: (dim2,elastic_psv), (dim2,acoustic), (dim3,elastic), (dim3,acoustic)
// — 4 entries

TEST(FindIn, LargerArray_FindDim3Elastic) {
  constexpr LargerET::combo_type search{ D::dim3, M::elastic };
  constexpr std::size_t idx = td::find_in_v<LargerET::combos, search>;
  // It must be ≥ 0 and < 4, and the combo at that index must match
  static_assert(idx < LargerET::size);
  static_assert(LargerET::combos[idx] == search);
  EXPECT_LT(idx, LargerET::size);
  EXPECT_TRUE(LargerET::combos[idx] == search);
}

TEST(FindIn, LargerArray_Dim2ElasticNotPresent) {
  constexpr LargerET::combo_type search{ D::dim2, M::elastic };
  using Result = td::find_in_t<LargerET::combos, search>;
  static_assert(std::is_void_v<Result>);
  EXPECT_TRUE((std::is_void_v<Result>));
}
