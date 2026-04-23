// Unit tests for specfem/tag_dispatch/storage.hpp
//
// Covers:
//   Storage<T, ET>         — homogeneous storage (all slots hold T)
//   TypedStorage<Tmpl, ET> — heterogeneous storage (slot for TagsType holds
//   Tmpl<TagsType>)
//
// Both the compile-time get<TagsType>() and the runtime get(query_tags...) are
// tested.

#include "specfem/tag_dispatch/element_combinations.hpp"
#include "specfem/tag_dispatch/storage.hpp"
#include "specfem/tags.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using D = specfem::element::dimension_tag;
using M = specfem::element::medium_tag;
using P = specfem::element::property_tag;
using A = specfem::element::attenuation_tag;
using B = specfem::element::boundary_tag;

namespace td = specfem::tag_dispatch;

// SmallET: 2 valid combos
//   combos[0] = (dim2, elastic_psv, isotropic, none, none)
//   combos[1] = (dim2, acoustic,    isotropic, none, none)
using SmallET = td::element_combinations<
    td::dimension_set<D::dim2>, td::medium_set<M::elastic_psv, M::acoustic>,
    td::property_set<P::isotropic>, td::attenuation_set<A::none>,
    td::boundary_set<B::none> >;

using ElPSV = specfem::tags::Tags<D::dim2, M::elastic_psv, P::isotropic,
                                  A::none, B::none>;
using Acoustic =
    specfem::tags::Tags<D::dim2, M::acoustic, P::isotropic, A::none, B::none>;

// Helper struct for TypedStorage tests
template <typename T> struct Typed {
  int val = 0;
};

// ── Storage<int, SmallET> ─────────────────────────────────────────────

TEST(Storage, SizeMatchesET) {
  static_assert(td::Storage<int, SmallET>::size == SmallET::size);
  constexpr std::size_t storage_size = td::Storage<int, SmallET>::size;
  EXPECT_EQ(storage_size, SmallET::size);
}

TEST(Storage, DefaultConstructed_ZeroValue) {
  td::Storage<int, SmallET> s;
  EXPECT_EQ(s.get<ElPSV>(), 0);
  EXPECT_EQ(s.get<Acoustic>(), 0);
}

TEST(Storage, GetSet_Roundtrip) {
  td::Storage<int, SmallET> s;
  s.get<ElPSV>() = 10;
  EXPECT_EQ(s.get<ElPSV>(), 10);
  EXPECT_EQ(s.get<Acoustic>(), 0); // other slot unaffected
}

TEST(Storage, GetReturnsReference) {
  td::Storage<int, SmallET> s;
  int &ref = s.get<ElPSV>();
  ref = 99;
  EXPECT_EQ(s.get<ElPSV>(), 99);
}

TEST(Storage, TwoSlots_IndependentWrites) {
  td::Storage<int, SmallET> s;
  s.get<ElPSV>() = 11;
  s.get<Acoustic>() = 22;
  EXPECT_EQ(s.get<ElPSV>(), 11);
  EXPECT_EQ(s.get<Acoustic>(), 22);
}

TEST(Storage, ConstGet_Works) {
  td::Storage<int, SmallET> s;
  s.get<ElPSV>() = 7;
  const td::Storage<int, SmallET> &cs = s;
  EXPECT_EQ(cs.get<ElPSV>(), 7);
}

TEST(Storage, InitializerConstructor_AllSlotsSet) {
  td::Storage<int, SmallET> s([&]<typename>() { return 42; });
  EXPECT_EQ(s.get<ElPSV>(), 42);
  EXPECT_EQ(s.get<Acoustic>(), 42);
}

// ── runtime get(query_tags...) const ──────────────────────────────────

TEST(Storage, RuntimeGet_ByAllTags) {
  td::Storage<int, SmallET> s;
  s.get<ElPSV>() = 5;
  const td::Storage<int, SmallET> &cs = s;
  const auto &val =
      cs.get(D::dim2, M::elastic_psv, P::isotropic, A::none, B::none);
  EXPECT_EQ(val, 5);
}

TEST(Storage, RuntimeGet_ByPartialTag_Medium) {
  td::Storage<int, SmallET> s;
  s.get<Acoustic>() = 3;
  const td::Storage<int, SmallET> &cs = s;
  const auto &val = cs.get(M::acoustic);
  EXPECT_EQ(val, 3);
}

TEST(Storage, RuntimeGet_Throws_OnNoMatch) {
  const td::Storage<int, SmallET> s;
  // B::stacey is not in SmallET (only B::none is)
  EXPECT_THROW(s.get(B::stacey), std::runtime_error);
}

TEST(Storage, RuntimeGet_Throws_OnImpossibleMedium) {
  const td::Storage<int, SmallET> s;
  // M::elastic is dim3-only, cannot appear in SmallET (dim2-only)
  EXPECT_THROW(s.get(M::elastic), std::runtime_error);
}

// ── TypedStorage<Typed, SmallET> ─────────────────────────────────────

TEST(TypedStorage, SizeMatchesET) {
  static_assert(td::TypedStorage<Typed, SmallET>::size == SmallET::size);
  constexpr std::size_t typed_size = td::TypedStorage<Typed, SmallET>::size;
  EXPECT_EQ(typed_size, SmallET::size);
}

TEST(TypedStorage, DefaultConstructed_ZeroVal) {
  td::TypedStorage<Typed, SmallET> s;
  EXPECT_EQ(s.get<ElPSV>().val, 0);
  EXPECT_EQ(s.get<Acoustic>().val, 0);
}

TEST(TypedStorage, GetReturnsCorrectType) {
  td::TypedStorage<Typed, SmallET> s;
  static_assert(std::is_same_v<decltype(s.get<ElPSV>()), Typed<ElPSV> &>);
  static_assert(std::is_same_v<decltype(s.get<Acoustic>()), Typed<Acoustic> &>);
  EXPECT_TRUE((std::is_same_v<decltype(s.get<ElPSV>()), Typed<ElPSV> &>));
}

TEST(TypedStorage, GetSet_Roundtrip) {
  td::TypedStorage<Typed, SmallET> s;
  s.get<ElPSV>().val = 5;
  EXPECT_EQ(s.get<ElPSV>().val, 5);
  EXPECT_EQ(s.get<Acoustic>().val, 0); // other slot unaffected
}

TEST(TypedStorage, InitializerConstructor) {
  td::TypedStorage<Typed, SmallET> s(
      [&]<typename T>() -> Typed<T> { return { 77 }; });
  EXPECT_EQ(s.get<ElPSV>().val, 77);
  EXPECT_EQ(s.get<Acoustic>().val, 77);
}

TEST(TypedStorage, ConstGet_Works) {
  td::TypedStorage<Typed, SmallET> s;
  s.get<ElPSV>().val = 9;
  const td::TypedStorage<Typed, SmallET> &cs = s;
  EXPECT_EQ(cs.get<ElPSV>().val, 9);
}
