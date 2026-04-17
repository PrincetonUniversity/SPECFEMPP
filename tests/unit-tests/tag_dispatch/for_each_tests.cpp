// Unit tests for specfem/tag_dispatch/for_each.hpp
//
// Covers:
//   for_each(element_combinations<...>, f)
//     — invokes f.operator()<TagsType>() exactly ET::size times, once per
//       valid element combination, and each TagsType correctly reflects the
//       combination's tag values.

#include "specfem/tag_dispatch/element_combinations.hpp"
#include "specfem/tag_dispatch/for_each.hpp"
#include "specfem/tag_dispatch/is_valid.hpp"
#include <cstddef>
#include <gtest/gtest.h>
#include <set>
#include <typeindex>

using D = specfem::element::dimension_tag;
using M = specfem::element::medium_tag;
using P = specfem::element::property_tag;
using A = specfem::element::attenuation_tag;
using B = specfem::element::boundary_tag;

namespace td = specfem::tag_dispatch;

// SmallET: 2 valid combos (elastic_psv and acoustic, both dim2)
using SmallET = td::element_combinations<
    td::dimension_set<D::dim2>, td::medium_set<M::elastic_psv, M::acoustic>,
    td::property_set<P::isotropic>, td::attenuation_set<A::none>,
    td::boundary_set<B::none> >;

// EmptyET: 0 valid combos (dim3 × elastic_psv is always invalid)
using EmptyET = td::element_combinations<td::dimension_set<D::dim3>,
                                         td::medium_set<M::elastic_psv> >;

// MultiET: several valid combos spanning two dimensions and two media
using MultiET = td::element_combinations<
    td::dimension_set<D::dim2, D::dim3>,
    td::medium_set<M::elastic_psv, M::acoustic, M::elastic>,
    td::property_set<P::isotropic, P::anisotropic> >;

// ── Invocation count ──────────────────────────────────────────────────

TEST(ForEach, CallCount_MatchesSize) {
  int count = 0;
  td::for_each(SmallET{}, [&]<typename>() { ++count; });
  EXPECT_EQ(count, static_cast<int>(SmallET::size));
}

TEST(ForEach, EmptyET_NoCalls) {
  static_assert(EmptyET::size == 0);
  int count = 0;
  td::for_each(EmptyET{}, [&]<typename>() { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(ForEach, MultiTagSet_CountMatchesSize) {
  int count = 0;
  td::for_each(MultiET{}, [&]<typename>() { ++count; });
  EXPECT_EQ(count, static_cast<int>(MultiET::size));
}

// ── TagsType correctness ──────────────────────────────────────────────

TEST(ForEach, TagsTypes_HaveCorrectDimension) {
  // SmallET is dim2-only; every TagsType must report dim2
  td::for_each(SmallET{}, [&]<typename TagsType>() {
    EXPECT_EQ(TagsType::dimension_tag, D::dim2);
  });
}

TEST(ForEach, TagsTypes_AllUniqueTypes) {
  // Each call supplies a distinct specialisation of Tags<...>
  std::set<std::type_index> seen;
  td::for_each(SmallET{}, [&]<typename TagsType>() {
    auto [it, inserted] = seen.insert(std::type_index(typeid(TagsType)));
    EXPECT_TRUE(inserted) << "Duplicate TagsType encountered";
  });
  EXPECT_EQ(seen.size(), SmallET::size);
}

TEST(ForEach, TagsTypes_AllValid) {
  // Re-assemble the combo from the static members of TagsType and verify it
  // passes the full validity predicate.
  using namespace specfem::tag_dispatch::impl;
  td::for_each(SmallET{}, [&]<typename TagsType>() {
    ElementTagTuple combo{ TagsType::dimension_tag, TagsType::medium_tag,
                           TagsType::property_tag, TagsType::attenuation_tag,
                           TagsType::boundary_tag };
    EXPECT_TRUE(is_valid_full_combo(combo));
  });
}

// ── Per-medium counting ───────────────────────────────────────────────

TEST(ForEach, CountByMedium_SumsToTotalSize) {
  // Count how many combos are elastic_psv vs acoustic; total must equal size.
  int psv_count = 0, ac_count = 0;
  td::for_each(SmallET{}, [&]<typename TagsType>() {
    if constexpr (TagsType::medium_tag == M::elastic_psv)
      ++psv_count;
    if constexpr (TagsType::medium_tag == M::acoustic)
      ++ac_count;
  });
  EXPECT_EQ(psv_count + ac_count, static_cast<int>(SmallET::size));
  EXPECT_EQ(psv_count, 1); // exactly one combo per medium in SmallET
  EXPECT_EQ(ac_count, 1);
}

TEST(ForEach, MultiET_Dim3CombosHaveNoneOrNoRestrictedBoundary) {
  // For any dim3 combo produced by for_each, dimension_tag must be dim3.
  // Not checking the boundary here because MultiET has no boundary_set —
  // this just verifies the dim3 medium correctly round-trips through Tags.
  int dim3_count = 0;
  td::for_each(MultiET{}, [&]<typename TagsType>() {
    if constexpr (TagsType::dimension_tag == D::dim3) {
      // dim3 media can only be elastic or acoustic (elastic_psv is 2D-only)
      EXPECT_TRUE(TagsType::medium_tag == M::elastic ||
                  TagsType::medium_tag == M::acoustic);
      ++dim3_count;
    }
  });
  EXPECT_GT(dim3_count, 0); // must have found at least one dim3 combo
}
