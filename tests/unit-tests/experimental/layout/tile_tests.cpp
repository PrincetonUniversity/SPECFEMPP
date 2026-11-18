#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define TILE_SUITE TileDebug
#define MAPPING_PROTOCOL_SUITE MappingProtocolDebug
#else
#define TILE_SUITE Tile
#define MAPPING_PROTOCOL_SUITE MappingProtocol
#endif
#include "restricted_mapping.hpp"
#include "specfem/experimental/layout/kokkos.hpp"
#include "specfem/experimental/layout/tile.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <set>
namespace hl = specfem::experimental::layout;
namespace specfem::testing::tile_impl {
using base = decltype(hl::make_layout(hl::make_tuple(8), hl::make_tuple(1)));
struct ShortSpan : base {
  ShortSpan() : base(hl::make_tuple(8), hl::make_tuple(1)) {}
  std::size_t required_span_size() const { return 7; }
  template <typename C> bool valid(const C &c) const {
    return base::valid(c) && base::offset(c) < 7;
  }
};
} // namespace specfem::testing::tile_impl
namespace specfem::testing::tile_impl {
template <int Case> struct SpanCase {
  static auto make() {
    const auto m = hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
    if constexpr (Case == 0)
      return hl::subview(ShortSpan{}, hl::all);
    else if constexpr (Case == 1)
      return hl::tile(m, 4);
    else if constexpr (Case == 2)
      return hl::permute<hl::ModePath<>, 0>(m);
    else if constexpr (Case == 3)
      return hl::subview(hl::tile(m, 4), hl::all);
    else
      return hl::tile(
          hl::permute<hl::ModePath<>, 0>(hl::subview(hl::tile(m, 4), hl::all)),
          2);
  }
};
} // namespace specfem::testing::tile_impl
template <typename Case>
class MAPPING_PROTOCOL_SUITE : public ::testing::Test {};
using span_cases = ::testing::Types<specfem::testing::tile_impl::SpanCase<0>,
                                    specfem::testing::tile_impl::SpanCase<1>,
                                    specfem::testing::tile_impl::SpanCase<2>,
                                    specfem::testing::tile_impl::SpanCase<3>,
                                    specfem::testing::tile_impl::SpanCase<4>>;
TYPED_TEST_SUITE(MAPPING_PROTOCOL_SUITE, span_cases);
TYPED_TEST(MAPPING_PROTOCOL_SUITE, SpanBoundsValidOffsets) {
  const auto mapping = TypeParam::make();
  EXPECT_LE(mapping.required_span_size(), 7u);
  for (int i = 0; i < 16; ++i) {
    const auto c = hl::make_tuple(i);
    if (mapping.valid(c)) {
      EXPECT_FALSE(mapping.empty());
      EXPECT_LT(mapping.offset(c), mapping.required_span_size());
    }
  }
}
TEST(TILE_SUITE, SpanProtocolChains) {
  const auto m = hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
  const auto t = hl::tile(m, 4);
  const auto s = hl::subview(t, hl::all);
  const auto p = hl::permute<hl::ModePath<>, 0>(s);
  const auto repeated = hl::tile(p, 2);
  const auto verify = [](const auto &mapping, const auto &parent) {
    EXPECT_LE(mapping.required_span_size(), parent.required_span_size());
    for (int i = 0; i < 16; ++i) {
      const auto c = hl::make_tuple(i);
      if (mapping.valid(c)) {
        EXPECT_FALSE(mapping.empty());
        EXPECT_LT(mapping.offset(c), mapping.required_span_size());
      }
    }
  };
  verify(t, m);
  verify(s, t);
  verify(p, s);
  verify(repeated, p);
}
TEST(TILE_SUITE, SliceLimitSurvives) {
  const auto m = hl::make_layout(hl::make_tuple(16), hl::make_tuple(1));
  const auto t = hl::tile(hl::subview(m, hl::range(0, 7)), hl::c<4>);
  EXPECT_EQ(t.envelope_size(), 8u);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(t.valid(hl::make_tuple(i)), i < 7);
    if (i < 7)
      EXPECT_EQ(t.offset(hl::make_tuple(i)), i);
  }
}
// The slice limit lives in the subview's predicate, not in an extent: a range
// with an unaligned end on a nested storage-tile mode rounds its envelope up.
// Tiling the lanes or the whole mode must keep the limit.
template <int Lanes> void expect_nested_slice_limit(int n) {
  constexpr int tiles = 3;
  constexpr int components = 2;
  const auto m = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<Lanes>, tiles), components),
      hl::make_tuple(hl::make_tuple(hl::c<1>, Lanes * components), Lanes));
  const auto s = hl::subview(m, hl::range(hl::c<0>, n), hl::all);
  const auto lanes = hl::tile(s, hl::make_tuple(hl::c<4>, hl::all), hl::all);
  const auto whole = hl::tile(s, hl::c<4>, hl::all);
  const auto envelope = static_cast<int>(whole.envelope_size()) / components;
  int valid = 0;
  for (int k = 0; k < components; ++k) {
    for (int i = 0; i < envelope; ++i) {
      const auto c = hl::make_tuple(i, k);
      const bool expected = i < n;
      EXPECT_EQ(lanes.valid(c), expected) << "n=" << n << " i=" << i;
      EXPECT_EQ(whole.valid(c), expected) << "n=" << n << " i=" << i;
      if (expected) {
        ++valid;
        const std::size_t offset =
            i % Lanes + Lanes * components * (i / Lanes) + Lanes * k;
        EXPECT_EQ(lanes.offset(c), offset);
        EXPECT_EQ(whole.offset(c), offset);
      }
    }
  }
  EXPECT_EQ(valid, n * components);
}

TEST(TILE_SUITE, SliceLimitSurvivesNestedRange) {
  for (const int n : { 13, 33, 65 }) {
    expect_nested_slice_limit<32>(n);
    expect_nested_slice_limit<64>(n);
  }
  expect_nested_slice_limit<8>(13);
}

TEST(TILE_SUITE, MultipleModes) {
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto t = hl::tile(m, 4, 2);
  const auto chained = hl::tile(hl::tile(m, 4, hl::all), hl::all, 2);
  std::set<std::size_t> offsets;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 8; ++i) {
      const auto c = hl::make_tuple(i, j);
      EXPECT_EQ(t.valid(c), i < 7 && j < 3);
      EXPECT_EQ(t.valid(c), chained.valid(c));
      if (t.valid(c)) {
        EXPECT_EQ(t.offset(c), i + 7 * j);
        EXPECT_EQ(t.offset(c), chained.offset(c));
        offsets.insert(t.offset(c));
      }
    }
  EXPECT_EQ(offsets.size(), 21u);
}
TEST(TILE_SUITE, CoordinateOracle) {
  const auto m = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<32>, hl::c<3>), hl::c<5>),
      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::c<160>), hl::c<32>));
  const auto t = hl::tile(m, hl::c<8>, hl::c<2>);
  static_assert(
      std::is_same_v<
          std::remove_cvref_t<decltype(hl::get<0>(hl::get<0>(t.shape())))>,
          std::integral_constant<int, 8>>);
  for (int a = 0; a < 96; ++a)
    for (int b = 0; b < 6; ++b) {
      const auto c = hl::make_tuple(a, b);
      EXPECT_EQ(t.valid(c), b < 5);
      if (b < 5)
        EXPECT_EQ(t.offset(c), a % 32 + 160 * (a / 32) + 32 * b);
    }
  // Tile the slow leaf c<3> by 2, a runtime size with a tail.
  const auto deep = hl::tile(m, hl::make_tuple(hl::all, 2), hl::all);
  EXPECT_EQ(deep.envelope_size(), 128u * 5u);
  for (int a = 0; a < 128; ++a) {
    for (int b = 0; b < 5; ++b) {
      const auto c = hl::make_tuple(a, b);
      EXPECT_EQ(deep.valid(c), a < 96);
      if (a < 96) {
        EXPECT_EQ(deep.offset(c),
                  static_cast<std::size_t>(a % 32 + 160 * (a / 32) + 32 * b));
      }
    }
  }
}
TEST(TILE_SUITE, StorageBoundaries) {
  for (int n : { 0, 1, 31, 32, 33, 63, 64, 65, 96, 128, 129 })
    for (int size : { 32, 64 }) {
      const auto m = hl::make_layout(hl::make_tuple(n), hl::make_tuple(1));
      const auto t = hl::tile(m, size);
      int count = 0;
      for (std::size_t i = 0; i < t.envelope_size(); ++i) {
        const auto c = hl::make_tuple(i);
        EXPECT_EQ(t.valid(c), i < static_cast<std::size_t>(n));
        if (t.valid(c)) {
          ++count;
          EXPECT_EQ(t.offset(c), i);
        }
      }
      EXPECT_EQ(count, n);
    }
}
TEST(TILE_SUITE, ComposesWithTransformations) {
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto t = hl::tile(m, 4, hl::all);
  const auto repeated = hl::tile(t, 2, hl::all);
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(t);
  const auto g = hl::group<hl::ModePath<>, 0, 2>(t);
  const auto u = hl::ungroup<hl::ModePath<0>>(t);
  const auto boundary = hl::subview(t, hl::make_tuple(hl::all, 1), hl::all);
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(repeated.valid(hl::make_tuple(i, j)), i < 7);
      if (i < 7) {
        EXPECT_EQ(repeated.offset(hl::make_tuple(i, j)),
                  static_cast<std::size_t>(i + 7 * j));
      }
      EXPECT_EQ(p.valid(hl::make_tuple(j, i)), i < 7);
      EXPECT_EQ(g.valid(hl::make_tuple(i + 8 * j)), i < 7);
      EXPECT_EQ(u.valid(hl::make_tuple(i % 4, i / 4, j)), i < 7);
      if (i < 7)
        EXPECT_EQ(p.offset(hl::make_tuple(j, i)), i + 7 * j);
    }
  EXPECT_FALSE(boundary.valid(hl::make_tuple(hl::make_tuple(3), 0)));
  EXPECT_EQ(boundary.offset(hl::make_tuple(hl::make_tuple(2), 2)), 20u);
}
TEST(TILE_SUITE, PreservesInheritedValidity) {
  const specfem::testing::layout_transform_impl::RestrictedMapping m;
  const auto t = hl::tile(m, hl::make_tuple(hl::all, 2), 2);
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 8; ++i) {
      const auto c = hl::make_tuple(i, j);
      const int offset = i + 6 * j;
      EXPECT_EQ(t.valid(c),
                i < 6 && offset != 1 && offset != 10 && offset != 11);
    }
}
TEST(TILE_SUITE, SpanStaysWithinValidOffsets) {
  double storage[7]{};
  const auto m = hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
  const auto v = hl::make_view(storage, 7, m);
  const auto t = hl::tile(v, 4);
  const auto s = hl::subview(t, hl::all);
  EXPECT_EQ(s.layout().required_span_size(), 7u);
  s(hl::make_tuple(6)) = 9;
  EXPECT_EQ(storage[6], 9);
}
TEST(TILE_SUITE, MaximumCount) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const auto m = hl::make_layout(hl::make_tuple(maximum), hl::make_tuple(1));
  const auto one = hl::tile(m, std::size_t{ 1 });
  const auto whole = hl::tile(m, maximum);
  EXPECT_EQ(one.envelope_size(), maximum);
  EXPECT_EQ(whole.envelope_size(), maximum);
  EXPECT_EQ(whole.offset(hl::make_tuple(maximum - 1)), maximum - 1);
}
TEST(TILE_SUITE, EmptyDomains) {
  const auto m =
      hl::make_layout(hl::make_tuple(0, 2),
                      hl::make_tuple(std::numeric_limits<std::size_t>::max(),
                                     std::size_t{ 1 }));
  const auto t = hl::tile(m, 4, 3);
  EXPECT_TRUE(t.empty());
  EXPECT_EQ(t.required_span_size(), 0u);
  EXPECT_FALSE(t.valid(hl::make_tuple(0, 0)));
  const auto scalar =
      hl::tile(hl::make_layout(hl::make_tuple(), hl::make_tuple()));
  EXPECT_TRUE(scalar.valid(hl::make_tuple()));
}
namespace specfem::testing::tile_impl {
void kernel_tiling();
} // namespace specfem::testing::tile_impl
void specfem::testing::tile_impl::kernel_tiling() {
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto captured = hl::tile(m, 4, 2);
  int errors = 0;
  Kokkos::parallel_reduce(
      "tile_oracle", 32,
      KOKKOS_LAMBDA(int k, int &bad) {
        const int i = k % 8, j = k / 8;
        const int size = 4;
        const auto local = hl::tile(m, size, 2);
        const auto c = hl::make_tuple(hl::make_tuple(i % 4, i / 4),
                                      hl::make_tuple(j % 2, j / 2));
        const bool valid = i < 7 && j < 3;
        if (local.valid(c) != valid || captured.valid(c) != valid)
          ++bad;
        if (valid &&
            (local.offset(c) != i + 7 * j || captured.offset(c) != i + 7 * j))
          ++bad;
      },
      errors);
  EXPECT_EQ(errors, 0);

  // Tile a device view inside the kernel and write through it.
  Kokkos::View<int *> backing("tile_kernel", 21);
  Kokkos::deep_copy(backing, -1);
  const auto view = hl::make_view(backing, m);
  Kokkos::parallel_for(
      "tile_view_kernel", 32, KOKKOS_LAMBDA(int k) {
        const int i = k % 8, j = k / 8;
        const auto tiled = hl::tile(view, 4, 2);
        const auto c = hl::make_tuple(i, j);
        if (tiled.layout().valid(c)) {
          tiled(c) = 10 * j + i;
        }
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, backing);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 7; ++i) {
      EXPECT_EQ(host(i + 7 * j), 10 * j + i);
    }
  }
}
TEST(TILE_SUITE, KernelTiling) { specfem::testing::tile_impl::kernel_tiling(); }
TEST(TILE_SUITE, AliasingAndLifetime) {
  const auto retained = [] {
    Kokkos::View<int *, Kokkos::HostSpace> native("retained_tile", 7);
    Kokkos::deep_copy(native, -1);
    const auto t = hl::tile(hl::make_view(native), 4);
    t(hl::make_tuple(6)) = 51;
    EXPECT_EQ(&t(hl::make_tuple(6)), &native(6));
    return t;
  }();
  Kokkos::View<int *, Kokkos::HostSpace> other("distractor", 7);
  Kokkos::deep_copy(other, -2);
  EXPECT_EQ(retained(hl::make_tuple(6)), 51);
  const int data[7]{};
  const auto m = hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
  const auto read_only = hl::tile(hl::make_view(data, 7, m), 4);
  static_assert(
      std::is_same_v<decltype(read_only(hl::make_tuple(0))), const int &>);
  EXPECT_EQ(&read_only(hl::make_tuple(6)), data + 6);
}
TEST(TILE_SUITE, DerivedDescriptorAccepted) {
  const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
  const auto t = hl::tile(m, 4, hl::all);
  EXPECT_EQ(t.required_span_size(), 21u);
  std::set<std::size_t> offsets;
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 8; ++i) {
      const auto c = hl::make_tuple(i, j);
      if (t.valid(c))
        offsets.insert(t.offset(c));
    }
  EXPECT_EQ(offsets.size(), 21u);
  EXPECT_EQ(*offsets.rbegin(), 20u);
}
TEST(TILE_SUITE, ModesAtDifferentDepths) {
  const auto m = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::make_tuple(4, 3), 2), 5),
      hl::make_tuple(hl::make_tuple(hl::make_tuple(1, 4), 12), 24));
  const auto t =
      hl::tile(m, hl::make_tuple(hl::make_tuple(hl::all, 2), hl::all), 3);
  for (int outer = 0; outer < 2; ++outer)
    for (int b = 0; b < 4; ++b)
      for (int a = 0; a < 4; ++a)
        for (int j = 0; j < 6; ++j) {
          const auto c =
              hl::make_tuple(hl::make_tuple(hl::make_tuple(a, b), outer), j);
          const bool valid = b < 3 && j < 5;
          EXPECT_EQ(t.valid(c), valid);
          if (valid)
            EXPECT_EQ(t.offset(c), a + 4 * b + 12 * outer + 24 * j);
        }
  const auto identity = hl::tile(m, hl::all, hl::all);
  EXPECT_EQ(identity.offset(hl::make_tuple(23, 4)), 119u);
  const auto empty_child =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::make_tuple(), 4)),
                      hl::make_tuple(hl::make_tuple(hl::make_tuple(), 1)));
  const auto e = hl::tile(empty_child, 2);
  EXPECT_EQ(e.offset(hl::make_tuple(3)), 3u);
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  if (argc == 2 && std::strcmp(argv[1], "--invalid-device-tile") == 0) {
    {
      Kokkos::View<std::size_t *> result("invalid_tile", 1);
      Kokkos::parallel_for(
          "invalid_tile", 1, KOKKOS_LAMBDA(int i) {
            const auto m =
                hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
            result(i) = hl::tile(m, 4).offset(hl::make_tuple(7));
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
