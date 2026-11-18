#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define SUBVIEW_SUITE SubviewDebug
#else
#define SUBVIEW_SUITE Subview
#endif
#include "restricted_mapping.hpp"
#include "specfem/experimental/layout/kokkos.hpp"
#include "specfem/experimental/layout/subview.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <type_traits>

namespace hl = specfem::experimental::layout;

// Offsets below are hand-written stride sums of the parent descriptor.
TEST(SUBVIEW_SUITE, SelectorsAreTupleElements) {
  static_assert(hl::TupleElement<hl::All>);
  static_assert(hl::TupleElement<decltype(hl::range(1, hl::c<3>))>);
  const auto selector = hl::make_tuple(hl::all, hl::range(1, 3), 2);
  static_assert(hl::tuple_size_v<decltype(selector)> == 3);
  static_assert(std::is_same_v<decltype(hl::get<0>(selector)), hl::All>);
  EXPECT_EQ(hl::get<1>(selector).end, 3);
}

TEST(SUBVIEW_SUITE, FixesAndKeepsTopLevelModes) {
  // 7 x 3 packed column-major array.
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto column = hl::subview(m, hl::all, 1);
  static_assert(decltype(column)::rank() == 1);
  EXPECT_EQ(column.displacement(), 7u);
  EXPECT_EQ(column.required_span_size(), 14u);
  for (int row = 0; row < 7; ++row) {
    EXPECT_TRUE(column.valid(hl::make_tuple(row)));
    EXPECT_EQ(column.offset(hl::make_tuple(row)), 7u + row);
  }
  EXPECT_FALSE(column.valid(hl::make_tuple(7)));

  const auto row = hl::subview(m, 4, hl::all);
  EXPECT_EQ(row.displacement(), 4u);
  for (int col = 0; col < 3; ++col) {
    EXPECT_EQ(row.offset(hl::make_tuple(col)), 4u + 7u * col);
  }

  const auto scalar = hl::subview(m, 5, 2);
  static_assert(decltype(scalar)::rank() == 0);
  EXPECT_TRUE(scalar.valid(hl::make_tuple()));
  EXPECT_EQ(scalar.offset(hl::make_tuple()), 19u);
  EXPECT_EQ(scalar.required_span_size(), 20u);
}

TEST(SUBVIEW_SUITE, KeepsLeafRanges) {
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto rows = hl::subview(m, hl::range(2, 7), 1);
  EXPECT_EQ(rows.displacement(), 9u);
  EXPECT_EQ(rows.extent<0>(), 5u);
  EXPECT_EQ(rows.required_span_size(), 14u);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(rows.valid(hl::make_tuple(i)));
    EXPECT_EQ(rows.offset(hl::make_tuple(i)), 9u + i);
  }
  EXPECT_FALSE(rows.valid(hl::make_tuple(5)));
  EXPECT_FALSE(rows.valid(hl::make_tuple(-1)));

  const auto block = hl::subview(m, hl::range(1, 3), hl::range(1, 3));
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(block.offset(hl::make_tuple(i, j)), 8u + i + 7u * j);
    }
  }
  EXPECT_EQ(block.required_span_size(), 17u);

  for (int b = 0; b <= 7; ++b) {
    const auto none = hl::subview(m, hl::range(b, b), hl::all);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(none.required_span_size(), 0u);
    EXPECT_FALSE(none.valid(hl::make_tuple(0, 0)));
  }
}

TEST(SUBVIEW_SUITE, RoundsNestedRangeEnvelope) {
  // Element mode of 3 tiles of 4 lanes (tile stride 10) and a component mode.
  const auto m = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3), 2),
                                 hl::make_tuple(hl::make_tuple(1, 10), 5));
  const auto s = hl::subview(m, hl::range(4, 10), hl::all);
  EXPECT_EQ(s.displacement(), 10u);
  EXPECT_EQ(hl::get<0>(hl::get<0>(s.shape())), 4);
  EXPECT_EQ(hl::get<1>(hl::get<0>(s.shape())), 2u);
  EXPECT_EQ(s.envelope_size(), 16u);
  std::set<std::size_t> offsets;
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 8; ++i) {
      const bool expected = i < 6;
      const auto scalar = hl::make_tuple(i, k);
      const auto nested = hl::make_tuple(hl::make_tuple(i % 4, i / 4), k);
      EXPECT_EQ(s.valid(scalar), expected);
      EXPECT_EQ(s.valid(nested), expected);
      if (expected) {
        const int e = i + 4;
        const std::size_t offset = e % 4 + 10 * (e / 4) + 5 * k;
        EXPECT_EQ(s.offset(scalar), offset);
        EXPECT_EQ(s.offset(nested), offset);
        EXPECT_TRUE(offsets.insert(offset).second);
        EXPECT_LT(offset, s.required_span_size());
      }
    }
  }
  EXPECT_LE(s.required_span_size(), m.required_span_size());

  const auto aligned = hl::subview(m, hl::range(4, 12), hl::all);
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(aligned.valid(hl::make_tuple(i, 1)));
  }
}

// The slowest leaf need not be the last child: a trailing leafless child is
// skipped, and the leaf can sit two levels down.
TEST(SUBVIEW_SUITE, RoundsNestedRangeSlowestLeaf) {
  const auto trailing = hl::make_layout(
      hl::make_tuple(hl::make_tuple(4, 3, hl::make_tuple()), 2),
      hl::make_tuple(hl::make_tuple(1, 10, hl::make_tuple()), 5));
  const auto t = hl::subview(trailing, hl::range(4, 10), hl::all);
  EXPECT_EQ(t.displacement(), 10u);
  EXPECT_EQ(hl::get<1>(hl::get<0>(t.shape())), 2u);
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(t.valid(hl::make_tuple(i, k)), i < 6);
      if (i < 6) {
        const int e = i + 4;
        EXPECT_EQ(t.offset(hl::make_tuple(i, k)),
                  static_cast<std::size_t>(e % 4 + 10 * (e / 4) + 5 * k));
      }
    }
  }

  // ((2, (3, 2)), 2) with strides ((1, (4, 16)), 40). The slowest leaf of the
  // first mode is the inner 2 with stride 16, so the block is 2 * 3 = 6.
  const auto deep = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<2>, hl::make_tuple(3, hl::c<2>)), 2),
      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::make_tuple(4, hl::c<16>)),
                     40));
  const auto d = hl::subview(deep, hl::range(6, 11), hl::all);
  EXPECT_EQ(d.displacement(), 16u);
  EXPECT_EQ(hl::get<1>(hl::get<1>(hl::get<0>(d.shape()))), 1u);
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(d.valid(hl::make_tuple(i, k)), i < 5);
      if (i < 5) {
        const int e = i + 6;
        const std::size_t offset =
            e % 2 + 4 * ((e / 2) % 3) + 16 * (e / 6) + 40 * k;
        EXPECT_EQ(d.offset(hl::make_tuple(i, k)), offset);
      }
    }
  }
}

template <int Tile>
void expect_storage_tile_tails(std::initializer_list<int> counts) {
  constexpr int tiles = 3;
  const auto m =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<Tile>, tiles)),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, Tile)));
  for (const int n : counts) {
    const auto s = hl::subview(m, hl::range(hl::c<0>, n));
    static_assert(
        std::is_same_v<
            std::remove_cvref_t<decltype(hl::get<0>(hl::get<0>(s.shape())))>,
            std::integral_constant<int, Tile>>);
    static_assert(std::is_same_v<decltype(s.displacement()),
                                 std::integral_constant<std::size_t, 0>>);
    EXPECT_EQ(hl::get<1>(hl::get<0>(s.shape())),
              static_cast<std::size_t>((n + Tile - 1) / Tile));
    int valid = 0;
    for (int i = 0; i < tiles * Tile; ++i) {
      if (s.valid(hl::make_tuple(i))) {
        ++valid;
        EXPECT_EQ(s.offset(hl::make_tuple(i)), static_cast<std::size_t>(i));
      }
    }
    EXPECT_EQ(valid, n);
  }
}

TEST(SUBVIEW_SUITE, StorageTileTails) {
  expect_storage_tile_tails<32>({ 0, 1, 31, 32, 33, 63, 64, 65, 96 });
  expect_storage_tile_tails<64>({ 0, 1, 63, 64, 65, 127, 128, 129, 192 });
}

TEST(SUBVIEW_SUITE, SelectsNestedParts) {
  const auto m = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3), 2),
                                 hl::make_tuple(hl::make_tuple(1, 10), 5));
  // One tile: the lane leaf stays inside a one-child tuple.
  const auto tile = hl::subview(m, hl::make_tuple(hl::all, 2), hl::all);
  static_assert(hl::tuple_size_v<decltype(hl::get<0>(tile.shape()))> == 1);
  EXPECT_EQ(tile.displacement(), 20u);
  for (int k = 0; k < 2; ++k) {
    for (int lane = 0; lane < 4; ++lane) {
      const std::size_t offset = lane + 20 + 5 * k;
      EXPECT_EQ(tile.offset(hl::make_tuple(lane, k)), offset);
      EXPECT_EQ(tile.offset(hl::make_tuple(hl::make_tuple(lane), k)), offset);
    }
  }
  EXPECT_FALSE(tile.valid(hl::make_tuple(4, 0)));

  // Lanes 1..2 of tile 1, component 1, in one call.
  const auto part = hl::subview(m, hl::make_tuple(hl::range(1, 3), 1), 1);
  static_assert(decltype(part)::rank() == 1);
  EXPECT_EQ(part.displacement(), 16u);
  EXPECT_EQ(part.offset(hl::make_tuple(0)), 16u);
  EXPECT_EQ(part.offset(hl::make_tuple(1)), 17u);
  EXPECT_FALSE(part.valid(hl::make_tuple(2)));

  // Fixing every part of a nested mode leaves an empty tuple of extent one.
  const auto point = hl::subview(m, hl::make_tuple(3, 1), hl::all);
  static_assert(hl::tuple_size_v<decltype(hl::get<0>(point.shape()))> == 0);
  EXPECT_EQ(point.offset(hl::make_tuple(0, 1)), 18u);
  EXPECT_EQ(point.offset(hl::make_tuple(hl::make_tuple(), 1)), 18u);

  // Depth three: ((2, (3, 2)), 2) with strides ((1, (4, 16)), 40).
  const auto deep = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<2>, hl::make_tuple(3, hl::c<2>)), 2),
      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::make_tuple(4, hl::c<16>)),
                     40));
  const auto inner = hl::subview(
      deep, hl::make_tuple(hl::all, hl::make_tuple(hl::range(1, 3), 1)),
      hl::all);
  for (int d = 0; d < 2; ++d) {
    for (int b = 0; b < 2; ++b) {
      for (int a = 0; a < 2; ++a) {
        const std::size_t offset = a + 4 * (b + 1) + 16 + 40 * d;
        EXPECT_EQ(inner.offset(
                      hl::make_tuple(hl::make_tuple(a, hl::make_tuple(b)), d)),
                  offset);
        EXPECT_EQ(inner.offset(hl::make_tuple(a + 2 * b, d)), offset);
      }
    }
  }
}

// Holes at original offsets 1, 10 and 11, with offset a + 2 b + 6 c.
KOKKOS_INLINE_FUNCTION bool hole_free(int offset) {
  return offset != 1 && offset != 10 && offset != 11;
}

TEST(SUBVIEW_SUITE, PreservesInheritedValidity) {
  const specfem::testing::layout_transform_impl::RestrictedMapping parent;

  // Keep b in [1, 3) through a nested part selector.
  const auto s =
      hl::subview(parent, hl::make_tuple(hl::all, hl::range(1, 3)), hl::all);
  // Repeated selection: fix c = 1 on the result.
  const auto s2 = hl::subview(s, hl::all, 1);
  // Scalar range [2, 5) on the element mode with c = 0: rounds to 4 slots.
  const auto r = hl::subview(parent, hl::range(2, 5), 0);
  // Reorganization on both sides of a subview.
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(s);
  const auto q = hl::subview(hl::permute<hl::ModePath<>, 1, 0>(parent), hl::all,
                             hl::make_tuple(hl::all, hl::range(1, 3)));
  for (int c = 0; c < 2; ++c) {
    for (int b = 0; b < 2; ++b) {
      for (int a = 0; a < 2; ++a) {
        const int offset = a + 2 * (b + 1) + 6 * c;
        const bool expected = hole_free(offset);
        const auto coordinate = hl::make_tuple(hl::make_tuple(a, b), c);
        EXPECT_EQ(s.valid(coordinate), expected);
        EXPECT_EQ(s.valid(hl::make_tuple(a + 2 * b, c)), expected);
        EXPECT_EQ(p.valid(hl::make_tuple(c, a + 2 * b)), expected);
        EXPECT_EQ(q.valid(hl::make_tuple(c, hl::make_tuple(a, b))), expected);
        if (expected) {
          EXPECT_EQ(s.offset(coordinate), static_cast<std::size_t>(offset));
          EXPECT_EQ(p.offset(hl::make_tuple(c, a + 2 * b)),
                    static_cast<std::size_t>(offset));
        }
        if (c == 1) {
          EXPECT_EQ(s2.valid(hl::make_tuple(a + 2 * b)), expected);
        }
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    const int element = i + 2;
    const bool expected = i < 3 && hole_free(element);
    EXPECT_EQ(r.valid(hl::make_tuple(i)), expected);
    if (expected) {
      EXPECT_EQ(r.offset(hl::make_tuple(i)), static_cast<std::size_t>(element));
    }
  }
  EXPECT_FALSE(s.valid(hl::make_tuple(4, 0)));
}

TEST(SUBVIEW_SUITE, EmptyAndScalar) {
  // Zero-extent mode: empty ranges and other selections stay empty.
  const auto zero = hl::make_layout(hl::make_tuple(0, 3), hl::make_tuple(1, 1));
  const auto z1 = hl::subview(zero, hl::range(0, 0), 1);
  const auto z2 = hl::subview(zero, hl::all, 2);
  EXPECT_TRUE(z1.empty());
  EXPECT_TRUE(z2.empty());
  EXPECT_EQ(z2.required_span_size(), 0u);
  EXPECT_FALSE(z2.valid(hl::make_tuple(0)));

  // A leafless node has extent one; only the range count can empty it.
  const auto unit = hl::make_layout(hl::make_tuple(hl::make_tuple(), 3),
                                    hl::make_tuple(hl::make_tuple(), 1));
  const auto kept = hl::subview(unit, hl::range(0, 1), hl::all);
  EXPECT_FALSE(kept.empty());
  EXPECT_TRUE(kept.valid(hl::make_tuple(0, 2)));
  EXPECT_EQ(kept.offset(hl::make_tuple(0, 2)), 2u);
  for (int b = 0; b <= 1; ++b) {
    const auto none = hl::subview(unit, hl::range(b, b), hl::all);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(none.required_span_size(), 0u);
    EXPECT_FALSE(none.valid(hl::make_tuple(0, 0)));
  }

  // A disabled parent stays disabled, including a rank-zero result.
  const auto disabled =
      hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(1, 2), false);
  const auto d1 = hl::subview(disabled, hl::all, 1);
  const auto d0 = hl::subview(disabled, 1, 1);
  EXPECT_TRUE(d1.empty());
  EXPECT_FALSE(d1.valid(hl::make_tuple(0)));
  EXPECT_TRUE(d0.empty());
  EXPECT_EQ(d0.required_span_size(), 0u);
  EXPECT_FALSE(d0.valid(hl::make_tuple()));

  // A rank-zero parent takes no selectors.
  const auto scalar = hl::make_layout(hl::make_tuple(), hl::make_tuple());
  const auto same = hl::subview(scalar);
  EXPECT_TRUE(same.valid(hl::make_tuple()));
  EXPECT_EQ(same.offset(hl::make_tuple()), 0u);
  EXPECT_EQ(same.required_span_size(), 1u);
}

// An empty parent needs no representable span, so its strides can be huge.
// Selecting from it computes no displacement and diagnoses nothing.
TEST(SUBVIEW_SUITE, EmptyParentSkipsDisplacement) {
  const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
  const auto disabled =
      hl::make_layout(hl::make_tuple(2, 4), hl::make_tuple(1, huge), false);
  const auto zero = hl::make_layout(hl::make_tuple(0, 4),
                                    hl::make_tuple(std::size_t{ 1 }, huge));
  const auto check = [](const auto &s, const auto &coordinate) {
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.required_span_size(), 0u);
    EXPECT_FALSE(s.valid(coordinate));
  };
  check(hl::subview(disabled, hl::all, 3), hl::make_tuple(0));
  check(hl::subview(disabled, hl::all, hl::range(3, 4)), hl::make_tuple(0, 0));
  check(hl::subview(zero, hl::all, 3), hl::make_tuple(0));
  check(hl::subview(zero, hl::all, hl::range(3, 4)), hl::make_tuple(0, 0));
}

TEST(SUBVIEW_SUITE, StaticMetadata) {
  constexpr auto m = hl::make_layout(hl::make_tuple(hl::c<7>, hl::c<3>),
                                     hl::make_tuple(hl::c<1>, hl::c<7>));
  constexpr auto s = hl::subview(m, hl::range(hl::c<2>, hl::c<7>), hl::c<1>);
  static_assert(std::is_same_v<decltype(s.displacement()),
                               std::integral_constant<std::size_t, 9>>);
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(hl::get<0>(s.shape()))>,
                     std::integral_constant<std::size_t, 5>>);
  static_assert(s.offset(hl::make_tuple(4)) == 13);
  using Access = specfem::experimental::layout_impl::LayoutAccess;
  constexpr auto descriptor =
      Access::derive(s.shape(), s.strides(), true, s.displacement());
  static_assert(
      std::is_same_v<decltype(descriptor.offset(hl::make_tuple(hl::c<4>))),
                     std::integral_constant<std::size_t, 13>>);
  constexpr auto disabled = Access::derive(
      hl::make_tuple(hl::c<2>),
      hl::make_tuple(hl::c<std::numeric_limits<std::size_t>::max()>), false);
  static_assert(disabled.required_span_size() == 0);
  static_assert(s.required_span_size() == 14);
  static_assert(!s.valid(hl::make_tuple(5)));
  // all keeps static leaves; a runtime begin makes the displacement runtime.
  const auto t = hl::subview(m, hl::all, 1);
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(hl::get<0>(t.shape()))>,
                     std::integral_constant<int, 7>>);
  static_assert(std::is_same_v<decltype(t.displacement()), std::size_t>);
  EXPECT_EQ(t.displacement(), 7u);
  const auto dynamic_descriptor =
      Access::derive(t.shape(), t.strides(), true, t.displacement());
  static_assert(std::is_same_v<decltype(dynamic_descriptor.offset(
                                   hl::make_tuple(hl::c<4>))),
                               std::size_t>);
  EXPECT_EQ(dynamic_descriptor.offset(hl::make_tuple(hl::c<4>)), 11u);
}

TEST(SUBVIEW_SUITE, MaximumDisplacement) {
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const auto m =
      hl::make_layout(hl::make_tuple(2), hl::make_tuple(maximum - 1));
  const auto fixed = hl::subview(m, 1);
  EXPECT_EQ(fixed.displacement(), maximum - 1);
  EXPECT_EQ(fixed.offset(hl::make_tuple()), maximum - 1);
  EXPECT_EQ(fixed.required_span_size(), maximum);
  const auto ranged = hl::subview(m, hl::range(1, 2));
  EXPECT_EQ(ranged.offset(hl::make_tuple(0)), maximum - 1);
  EXPECT_EQ(ranged.required_span_size(), maximum);
}

TEST(SUBVIEW_SUITE, AliasingAndLifetime) {
  const auto retained = [] {
    Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace> native(
        "retained", 7, 3);
    Kokkos::deep_copy(native, -1);
    const auto column = hl::subview(hl::make_view(native), hl::range(2, 7), 1);
    column(hl::make_tuple(1)) = 51;
    EXPECT_EQ(&native(3, 1), &column(hl::make_tuple(1)));
    EXPECT_EQ(native(3, 1), 51);
    EXPECT_EQ(native(2, 1), -1);
    return column;
  }();
  // Allocate a distractor after every original handle has left scope.
  Kokkos::View<int *, Kokkos::HostSpace> other("other", 21);
  Kokkos::deep_copy(other, -2);
  EXPECT_EQ(retained(hl::make_tuple(1)), 51);
  retained(hl::make_tuple(4)) = 73;
  EXPECT_EQ(retained(hl::make_tuple(4)), 73);

  int data[20];
  for (auto &value : data) {
    value = -1;
  }
  const auto mapping = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3)),
                                       hl::make_tuple(hl::make_tuple(1, 8)));
  const auto raw = hl::make_view(data, 20, mapping);
  const auto tile = hl::subview(raw, hl::make_tuple(hl::all, 2));
  tile(hl::make_tuple(3)) = 19;
  EXPECT_EQ(&tile(hl::make_tuple(3)), data + 19);
  EXPECT_EQ(data[19], 19);
  const auto read_only =
      hl::subview(hl::make_view(static_cast<const int *>(data), 20, mapping),
                  hl::range(4, 6));
  static_assert(
      std::is_same_v<decltype(read_only(hl::make_tuple(0))), const int &>);
  EXPECT_EQ(&read_only(hl::make_tuple(1)), data + 9);
}

namespace specfem::testing::layout_subview_impl {
void expect_kernel_selection();
} // namespace specfem::testing::layout_subview_impl

void specfem::testing::layout_subview_impl::expect_kernel_selection() {
  // Element mode of 3 tiles of 4 lanes, then a component mode of 2.
  constexpr int elements = 10;
  Kokkos::View<int *> backing("subview_kernel", 24);
  Kokkos::deep_copy(backing, -1);
  const auto mapping =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<4>, 3), 2),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 8), 4));
  const auto captured = hl::subview(hl::make_view(backing, mapping),
                                    hl::range(hl::c<0>, elements), hl::all);
  Kokkos::parallel_for(
      "subview_kernel", 24, KOKKOS_LAMBDA(int i) {
        const int lane = i % 4;
        const int tile = (i / 4) % 3;
        const int component = i / 12;
        const auto one_tile =
            hl::subview(captured, hl::make_tuple(hl::all, tile), component);
        const auto q = hl::make_tuple(lane);
        if (one_tile.layout().valid(q)) {
          one_tile(q) = 100 * component + 4 * tile + lane;
        }
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, backing);
  for (int component = 0; component < 2; ++component) {
    for (int element = 0; element < 12; ++element) {
      const int offset = element % 4 + 8 * (element / 4) + 4 * component;
      EXPECT_EQ(host(offset),
                element < elements ? 100 * component + element : -1);
    }
  }
}

TEST(SUBVIEW_SUITE, KernelSelection) {
  specfem::testing::layout_subview_impl::expect_kernel_selection();
}

TEST(SUBVIEW_SUITE, ComposesWithReorganization) {
  const auto m =
      hl::make_layout(hl::make_tuple(2, 3, 4), hl::make_tuple(1, 2, 6));
  const auto s = hl::subview(m, hl::all, 1, hl::all);
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(s);
  const auto g = hl::group<hl::ModePath<>, 0, 2>(s);
  const auto back =
      hl::subview(hl::permute<hl::ModePath<>, 2, 0, 1>(m), hl::all, hl::all, 1);
  EXPECT_EQ(p.displacement(), 2u);
  EXPECT_EQ(p.required_span_size(), s.required_span_size());
  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 2; ++i) {
      const std::size_t expected = i + 2 * 1 + 6 * k;
      EXPECT_EQ(p.offset(hl::make_tuple(k, i)), expected);
      EXPECT_EQ(g.offset(hl::make_tuple(i + 2 * k)), expected);
      EXPECT_EQ(back.offset(hl::make_tuple(k, i)), expected);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  if (argc == 2 && std::strcmp(argv[1], "--invalid-device-selection") == 0) {
    {
      Kokkos::View<std::size_t *> result("invalid_result", 1);
      Kokkos::parallel_for(
          "invalid_subview", 1, KOKKOS_LAMBDA(int i) {
            const auto parent =
                hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3), 2),
                                hl::make_tuple(hl::make_tuple(1, 10), 5));
            // Scalar 6 lies in the rounded envelope (8) but not in [0, 6).
            const auto s = hl::subview(parent, hl::range(4, 10), hl::all);
            result(i) = s.offset(hl::make_tuple(i + 6, 0));
          });
      Kokkos::fence();
    }
    Kokkos::finalize();
    return 0;
  }
  const int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
