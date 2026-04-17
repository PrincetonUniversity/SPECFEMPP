// Unit tests for specfem/tag_dispatch/element_combinations.hpp
//
// Covers:
//   - element_combinations<...>::size  (constexpr count of valid combos)
//   - element_combinations<...>::combos (compile-time array of valid combos)
//   - operator* tag-set composition
//   - Zero-size result when no combo is valid
//   - No-duplicates invariant across the whole combo list

#include "specfem/tag_dispatch/element_combinations.hpp"
#include "specfem/tag_dispatch/is_valid.hpp"
#include <cstddef>
#include <gtest/gtest.h>

using D = specfem::element::dimension_tag;
using M = specfem::element::medium_tag;
using P = specfem::element::property_tag;
using A = specfem::element::attenuation_tag;
using B = specfem::element::boundary_tag;

namespace td = specfem::tag_dispatch;

// ── Shared type aliases ───────────────────────────────────────────────

// 2-arity: (dim2|dim3) × (elastic_psv|elastic|acoustic)
// Valid combos: (dim2,elastic_psv), (dim2,acoustic), (dim3,elastic),
// (dim3,acoustic) → 4
using DimMed = td::element_combinations<
    td::dimension_set<D::dim2, D::dim3>,
    td::medium_set<M::elastic_psv, M::elastic, M::acoustic> >;

// 5-arity minimal set → exactly 2 combos (one per medium)
using SmallET = td::element_combinations<
    td::dimension_set<D::dim2>, td::medium_set<M::elastic_psv, M::acoustic>,
    td::property_set<P::isotropic>, td::attenuation_set<A::none>,
    td::boundary_set<B::none> >;

// Impossible: dim3 × elastic_psv is always invalid → size 0
using EmptyET = td::element_combinations<td::dimension_set<D::dim3>,
                                         td::medium_set<M::elastic_psv> >;

// All known tag values (5-arity)
using FullAllTags = td::element_combinations<
    td::dimension_set<D::dim2, D::dim3>,
    td::medium_set<M::elastic_psv, M::elastic_sh, M::elastic_psv_t, M::acoustic,
                   M::poroelastic, M::electromagnetic_te, M::elastic,
                   M::elastic_spin>,
    td::property_set<P::isotropic, P::anisotropic, P::isotropic_cosserat>,
    td::attenuation_set<A::none, A::constant_isotropic>,
    td::boundary_set<B::none, B::acoustic_free_surface, B::stacey,
                     B::composite_stacey_dirichlet> >;

// ── DimMed: 2-arity tests ─────────────────────────────────────────────

TEST(ElementCombinations, TwoTagSets_Size) {
  static_assert(DimMed::size == 4);
  EXPECT_EQ(DimMed::size, std::size_t(4));
}

TEST(ElementCombinations, TwoTagSets_AllValid) {
  for (auto const &combo : DimMed::combos)
    EXPECT_TRUE(specfem::tag_dispatch::impl::is_valid(combo));
}

TEST(ElementCombinations, TwoTagSets_NoDuplicates) {
  for (std::size_t i = 0; i < DimMed::size; ++i)
    for (std::size_t j = i + 1; j < DimMed::size; ++j)
      EXPECT_FALSE(DimMed::combos[i] == DimMed::combos[j])
          << "Duplicate entries at indices " << i << " and " << j;
}

TEST(ElementCombinations, TwoTagSets_ExpectedContent) {
  bool found_d2_psv = false, found_d2_ac = false;
  bool found_d3_el = false, found_d3_ac = false;
  for (auto const &c : DimMed::combos) {
    auto d = c.get<0>();
    auto m = c.get<1>();
    if (d == D::dim2 && m == M::elastic_psv)
      found_d2_psv = true;
    if (d == D::dim2 && m == M::acoustic)
      found_d2_ac = true;
    if (d == D::dim3 && m == M::elastic)
      found_d3_el = true;
    if (d == D::dim3 && m == M::acoustic)
      found_d3_ac = true;
    // (dim2, elastic) must NOT be present
    EXPECT_FALSE(d == D::dim2 && m == M::elastic)
        << "(dim2, elastic) is not a valid 2D medium";
    // (dim3, elastic_psv) must NOT be present
    EXPECT_FALSE(d == D::dim3 && m == M::elastic_psv)
        << "(dim3, elastic_psv) is not a valid 3D medium";
  }
  EXPECT_TRUE(found_d2_psv);
  EXPECT_TRUE(found_d2_ac);
  EXPECT_TRUE(found_d3_el);
  EXPECT_TRUE(found_d3_ac);
}

// ── SmallET: 5-arity minimal tests ───────────────────────────────────

TEST(ElementCombinations, FiveTagSets_Size) {
  static_assert(SmallET::size == 2);
  EXPECT_EQ(SmallET::size, std::size_t(2));
}

TEST(ElementCombinations, FiveTagSets_BothHaveDim2) {
  for (auto const &c : SmallET::combos)
    EXPECT_EQ(c.get<0>(), D::dim2);
}

TEST(ElementCombinations, FiveTagSets_DistinctMedia) {
  EXPECT_NE(SmallET::combos[0].get<1>(), SmallET::combos[1].get<1>());
}

TEST(ElementCombinations, FiveTagSets_AllValid) {
  for (auto const &c : SmallET::combos)
    EXPECT_TRUE(specfem::tag_dispatch::impl::is_valid(c));
}

// ── EmptyET ───────────────────────────────────────────────────────────

TEST(ElementCombinations, ImpossibleCombo_ZeroSize) {
  static_assert(EmptyET::size == 0);
  EXPECT_EQ(EmptyET::size, std::size_t(0));
}

// ── operator* composition ─────────────────────────────────────────────

TEST(ElementCombinations, OperatorStar_TwoSets_SameSizeAsDirectConstruct) {
  using Via_Star =
      decltype(td::dimension_set<D::dim2, D::dim3>{} *
               td::medium_set<M::elastic_psv, M::elastic, M::acoustic>{});
  static_assert(Via_Star::size == DimMed::size);
  EXPECT_EQ(Via_Star::size, DimMed::size);
}

TEST(ElementCombinations, OperatorStar_Extension_SameSizeAsDirectConstruct) {
  using Base =
      td::element_combinations<td::dimension_set<D::dim2>,
                               td::medium_set<M::elastic_psv, M::acoustic> >;
  using Extended = decltype(Base{} * td::property_set<P::isotropic>{});
  using Direct =
      td::element_combinations<td::dimension_set<D::dim2>,
                               td::medium_set<M::elastic_psv, M::acoustic>,
                               td::property_set<P::isotropic> >;
  static_assert(Extended::size == Direct::size);
  EXPECT_EQ(Extended::size, Direct::size);
}

TEST(ElementCombinations, OperatorStar_CombosMatch) {
  using Via_Star =
      decltype(td::dimension_set<D::dim2>{} *
               td::medium_set<M::elastic_psv, M::acoustic>{} *
               td::property_set<P::isotropic>{} *
               td::attenuation_set<A::none>{} * td::boundary_set<B::none>{});
  static_assert(Via_Star::size == SmallET::size);
  for (std::size_t i = 0; i < Via_Star::size; ++i)
    EXPECT_TRUE(Via_Star::combos[i] == SmallET::combos[i]);
}

// ── FullAllTags ───────────────────────────────────────────────────────

TEST(ElementCombinations, FullAllTags_SizePositive) {
  EXPECT_GT(FullAllTags::size, std::size_t(0));
}

TEST(ElementCombinations, FullAllTags_AllValid) {
  for (auto const &c : FullAllTags::combos)
    EXPECT_TRUE(specfem::tag_dispatch::impl::is_valid(c));
}

TEST(ElementCombinations, FullAllTags_NoDuplicates) {
  for (std::size_t i = 0; i < FullAllTags::size; ++i)
    for (std::size_t j = i + 1; j < FullAllTags::size; ++j)
      EXPECT_FALSE(FullAllTags::combos[i] == FullAllTags::combos[j])
          << "Duplicate entries at indices " << i << " and " << j;
}
