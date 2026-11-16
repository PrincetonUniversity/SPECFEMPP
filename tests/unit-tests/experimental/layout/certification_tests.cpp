// Certification runs only in debug builds and constant evaluation, so this
// suite forces debug checks even when the surrounding preset defines NDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "specfem/datatype/chunk_element_view.hpp"
#include "specfem/datatype/domain_view.hpp"
#include "specfem/datatype/element_view.hpp"
#include "specfem/experimental/layout/layout.hpp"
#include "specfem/experimental/layout/view.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace hl = specfem::experimental::layout;

namespace specfem::testing::layout_certification_impl {
struct FlatLeaf {
  std::size_t extent;
  std::size_t stride;
};

// Independent oracle: enumerate every envelope coordinate of a flattened
// layout and report whether two coordinates share an offset.
bool collides(const std::vector<FlatLeaf> &leaves);

// Compare every coordinate of a host Kokkos view with a layout describing it.
template <typename NativeView, typename Mapping>
void expect_matches_native(const NativeView &native, const Mapping &layout);

// Describe a native view by its runtime extents and strides.
template <typename NativeView> auto runtime_layout(const NativeView &native);

// Compact static strides: LayoutRight when Right, otherwise LayoutLeft.
template <bool Right, std::size_t I, std::size_t... Extents>
constexpr std::size_t compact_stride();

template <bool Right, std::size_t... Extents> constexpr auto static_layout();

// GPU domain storage: element tiles of TileSize lanes, then the remaining
// axes in LayoutLeft order inside each tile, then the tiles.
template <std::size_t TileSize, std::size_t Rank>
auto domain_tile_layout(const std::size_t (&extents)[Rank]);

template <specfem::element::dimension_tag DimensionTag, std::size_t TileSize,
          std::size_t Rank>
void expect_domain_tile_matches(const std::size_t (&extents)[Rank]);

// Every DomainView rank in data_access/container.hpp for one storage tile.
template <std::size_t TileSize> void expect_domain_tile_family(std::size_t n);

// Certify, construct, and write through accepted layouts inside a kernel.
void write_certified_on_device(int *tiled, int *strided, std::size_t elements);
} // namespace specfem::testing::layout_certification_impl

bool specfem::testing::layout_certification_impl::collides(
    const std::vector<FlatLeaf> &leaves) {
  std::size_t count = 1;
  for (const auto &leaf : leaves) {
    count *= leaf.extent;
  }
  std::set<std::size_t> offsets;
  for (std::size_t linear = 0; linear < count; ++linear) {
    std::size_t remaining = linear;
    std::size_t offset = 0;
    for (const auto &leaf : leaves) {
      offset += (remaining % leaf.extent) * leaf.stride;
      remaining /= leaf.extent;
    }
    if (!offsets.insert(offset).second) {
      return true;
    }
  }
  return false;
}

namespace specfem::testing::layout_certification_impl {
template <typename NativeView, typename Mapping, std::size_t... I>
void expect_matches_native_impl(const NativeView &native, const Mapping &layout,
                                std::index_sequence<I...>);
template <typename NativeView, std::size_t... I>
auto runtime_layout_impl(const NativeView &native, std::index_sequence<I...>);
template <bool Right, std::size_t... Extents, std::size_t... I>
constexpr auto static_layout_impl(std::index_sequence<I...>);
template <std::size_t TileSize, std::size_t Rank, std::size_t... I>
auto domain_tile_layout_impl(const std::size_t (&extents)[Rank],
                             std::index_sequence<I...>);
} // namespace specfem::testing::layout_certification_impl

template <typename NativeView, typename Mapping, std::size_t... I>
void specfem::testing::layout_certification_impl::expect_matches_native_impl(
    const NativeView &native, const Mapping &layout,
    std::index_sequence<I...>) {
  constexpr std::size_t rank = sizeof...(I);
  std::size_t count = 1;
  for (std::size_t r = 0; r < rank; ++r) {
    count *= native.extent(r);
  }
  ASSERT_LE(layout.required_span_size(), native.span());
  std::set<std::size_t> offsets;
  for (std::size_t linear = 0; linear < count; ++linear) {
    std::size_t index[rank == 0 ? 1 : rank]{};
    std::size_t remaining = linear;
    for (std::size_t r = 0; r < rank; ++r) {
      index[r] = remaining % native.extent(r);
      remaining /= native.extent(r);
    }
    const auto expected =
        static_cast<std::size_t>(&native(index[I]...) - native.data());
    ASSERT_EQ(layout.offset(hl::make_tuple(index[I]...)), expected);
    ASSERT_LT(expected, layout.required_span_size());
    ASSERT_TRUE(offsets.insert(expected).second);
  }
}

template <typename NativeView, typename Mapping>
void specfem::testing::layout_certification_impl::expect_matches_native(
    const NativeView &native, const Mapping &layout) {
  expect_matches_native_impl(native, layout,
                             std::make_index_sequence<NativeView::rank()>{});
}

template <typename NativeView, std::size_t... I>
auto specfem::testing::layout_certification_impl::runtime_layout_impl(
    const NativeView &native, std::index_sequence<I...>) {
  return hl::make_layout(
      hl::make_tuple(static_cast<std::size_t>(native.extent(I))...),
      hl::make_tuple(static_cast<std::size_t>(native.stride(I))...));
}

template <typename NativeView>
auto specfem::testing::layout_certification_impl::runtime_layout(
    const NativeView &native) {
  return runtime_layout_impl(native,
                             std::make_index_sequence<NativeView::rank()>{});
}

template <bool Right, std::size_t I, std::size_t... Extents>
constexpr std::size_t
specfem::testing::layout_certification_impl::compact_stride() {
  constexpr std::size_t extent[] = { Extents... };
  constexpr std::size_t rank = sizeof...(Extents);
  std::size_t stride = 1;
  // A LayoutRight stride multiplies the extents after axis I, a LayoutLeft
  // stride the extents before it. The != test avoids comparing r < 0.
  constexpr std::size_t first = Right ? I + 1 : 0;
  constexpr std::size_t last = Right ? rank : I;
  for (std::size_t r = first; r != last; ++r) {
    stride *= extent[r];
  }
  return stride;
}

template <bool Right, std::size_t... Extents, std::size_t... I>
constexpr auto specfem::testing::layout_certification_impl::static_layout_impl(
    std::index_sequence<I...>) {
  // Fully static descriptors certify at compile time.
  return hl::make_layout(
      hl::make_tuple(hl::c<Extents>...),
      hl::make_tuple(hl::c<compact_stride<Right, I, Extents...>()>...));
}

template <bool Right, std::size_t... Extents>
constexpr auto specfem::testing::layout_certification_impl::static_layout() {
  return static_layout_impl<Right, Extents...>(
      std::make_index_sequence<sizeof...(Extents)>{});
}

template <std::size_t TileSize, std::size_t Rank, std::size_t... I>
auto specfem::testing::layout_certification_impl::domain_tile_layout_impl(
    const std::size_t (&extents)[Rank], std::index_sequence<I...>) {
  // Stride of axis I + 1 inside a tile, and the size of one tile.
  std::size_t inner[Rank]{};
  inner[0] = TileSize;
  for (std::size_t r = 1; r < Rank; ++r) {
    inner[r] = inner[r - 1] * extents[r];
  }
  const std::size_t tiles =
      extents[0] / TileSize + (extents[0] % TileSize != 0);
  return hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<TileSize>, tiles), extents[I + 1]...),
      hl::make_tuple(hl::make_tuple(hl::c<std::size_t{ 1 }>, inner[Rank - 1]),
                     inner[I]...));
}

template <std::size_t TileSize, std::size_t Rank>
auto specfem::testing::layout_certification_impl::domain_tile_layout(
    const std::size_t (&extents)[Rank]) {
  return domain_tile_layout_impl<TileSize>(
      extents, std::make_index_sequence<Rank - 1>{});
}

template <specfem::element::dimension_tag DimensionTag, std::size_t TileSize,
          std::size_t Rank>
void specfem::testing::layout_certification_impl::expect_domain_tile_matches(
    const std::size_t (&extents)[Rank]) {
  using extents_type = Kokkos::dextents<std::size_t, Rank>;
  using production_mapping = typename specfem::datatype::DomainViewMapping<
      DimensionTag, TileSize>::template mapping<extents_type>;
  const auto production = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return production_mapping(extents_type{ extents[I]... });
  }(std::make_index_sequence<Rank>{});
  const auto layout = domain_tile_layout<TileSize>(extents);
  ASSERT_EQ(layout.required_span_size(), production.required_span_size());

  // Compare the whole envelope, including padded lanes of the last tile.
  const std::size_t lanes =
      (extents[0] / TileSize + (extents[0] % TileSize != 0)) * TileSize;
  std::size_t count = lanes;
  for (std::size_t r = 1; r < Rank; ++r) {
    count *= extents[r];
  }
  std::set<std::size_t> offsets;
  for (std::size_t linear = 0; linear < count; ++linear) {
    std::size_t index[Rank]{};
    std::size_t remaining = linear;
    index[0] = remaining % lanes;
    remaining /= lanes;
    for (std::size_t r = 1; r < Rank; ++r) {
      index[r] = remaining % extents[r];
      remaining /= extents[r];
    }
    const auto expected = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return production(index[I]...);
    }(std::make_index_sequence<Rank>{});
    const auto coordinate = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return hl::make_tuple(index[0], index[I + 1]...);
    }(std::make_index_sequence<Rank - 1>{});
    ASSERT_EQ(layout.offset(coordinate), expected);
    ASSERT_TRUE(offsets.insert(expected).second);
  }
}

void specfem::testing::layout_certification_impl::write_certified_on_device(
    int *tiled, int *strided, std::size_t elements) {
  Kokkos::parallel_for(
      "layout_certification_device", 1, KOKKOS_LAMBDA(int) {
        // GPU domain tile of 32 lanes over (elements, 5, 5).
        const std::size_t tiles = elements / 32 + (elements % 32 != 0);
        const auto tile_layout = hl::make_layout(
            hl::make_tuple(hl::make_tuple(hl::c<32>, tiles), hl::c<5>,
                           hl::c<5>),
            hl::make_tuple(hl::make_tuple(hl::c<1>, std::size_t{ 800 }),
                           hl::c<32>, hl::c<160>));
        const auto tile_view = hl::make_view(tiled, tiles * 800, tile_layout);
        for (std::size_t i = 0; i < elements; ++i) {
          for (int z = 0; z < 5; ++z) {
            for (int x = 0; x < 5; ++x) {
              tile_view(hl::make_tuple(i, z, x)) =
                  static_cast<int>(i * 25 + z * 5 + x);
            }
          }
        }
        // Fixed leading index of a LayoutLeft (3, 8, 5) buffer.
        const auto slot_layout = hl::make_layout(
            hl::make_tuple(std::size_t{ 8 }, std::size_t{ 5 }),
            hl::make_tuple(std::size_t{ 3 }, std::size_t{ 24 }));
        const auto slot_view = hl::make_view(strided + 1, 118, slot_layout);
        for (int j = 0; j < 8; ++j) {
          for (int k = 0; k < 5; ++k) {
            slot_view(hl::make_tuple(j, k)) = j + 8 * k;
          }
        }
      });
  Kokkos::fence();
}

template <std::size_t TileSize>
void specfem::testing::layout_certification_impl::expect_domain_tile_family(
    std::size_t n) {
  using specfem::element::dimension_tag;
  const std::size_t dim2_scalar[3] = { n, 5, 5 };
  const std::size_t dim2_vector[4] = { n, 5, 5, 2 };
  const std::size_t dim2_tensor[5] = { n, 5, 5, 2, 2 };
  const std::size_t dim3_scalar[4] = { n, 3, 3, 3 };
  const std::size_t dim3_vector[5] = { n, 3, 3, 3, 3 };
  const std::size_t dim3_tensor[6] = { n, 3, 3, 3, 3, 2 };
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim2, TileSize>(dim2_scalar)));
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim2, TileSize>(dim2_vector)));
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim2, TileSize>(dim2_tensor)));
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim3, TileSize>(dim3_scalar)));
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim3, TileSize>(dim3_vector)));
  ASSERT_NO_FATAL_FAILURE(
      (expect_domain_tile_matches<dimension_tag::dim3, TileSize>(dim3_tensor)));
}

namespace cert = specfem::testing::layout_certification_impl;

TEST(HierarchicalLayoutCertificationDebug, RejectsPartialOverlap) {
  // (2, 0) and (0, 1) both reach offset 2.
  ASSERT_TRUE(cert::collides({ { 4, 1 }, { 2, 2 } }));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(4, 2), hl::make_tuple(1, 2));
      },
      "uniqueness not certified");
}

TEST(HierarchicalLayoutCertificationDebug, RejectsInsideHostKernel) {
  const int block_stride = 3;
  EXPECT_DEATH(
      {
        Kokkos::parallel_for(
            "layout_certification_reject",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 1),
            [=](int) {
              [[maybe_unused]] const auto checked_result = hl::make_layout(
                  hl::make_tuple(4, 3), hl::make_tuple(1, block_stride));
            });
        Kokkos::fence();
      },
      "uniqueness not certified");
}

TEST(HierarchicalLayoutCertificationDebug, RejectsBroadcast) {
  ASSERT_TRUE(cert::collides({ { 3, 1 }, { 2, 0 } }));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(3, 2), hl::make_tuple(1, 0));
      },
      "layout broadcast");
}

TEST(HierarchicalLayoutCertificationDebug, RejectsEqualStrides) {
  ASSERT_TRUE(cert::collides({ { 2, 4 }, { 3, 4 } }));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(4, 4));
      },
      "uniqueness not certified");
}

TEST(HierarchicalLayoutCertificationDebug, RejectsNestedOverlapAtDepthThree) {
  // The innermost pair (2, 2) : (1, 1) collides inside an otherwise valid map.
  ASSERT_TRUE(cert::collides({ { 4, 1 }, { 2, 8 }, { 2, 8 }, { 3, 32 } }));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result = hl::make_layout(
            hl::make_tuple(hl::make_tuple(4, hl::make_tuple(2, 2)), 3),
            hl::make_tuple(hl::make_tuple(1, hl::make_tuple(8, 8)), 32));
      },
      "uniqueness not certified");
}

TEST(HierarchicalLayoutCertificationDebug, RejectsMixedStaticRuntimeOverlap) {
  // A static lane mode overlaps a runtime block stride smaller than its span.
  ASSERT_TRUE(cert::collides({ { 4, 1 }, { 3, 3 } }));
  const int block_stride = 3;
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result = hl::make_layout(
            hl::make_tuple(hl::make_tuple(hl::c<4>, 3)),
            hl::make_tuple(hl::make_tuple(hl::c<1>, block_stride)));
      },
      "uniqueness not certified");
}

TEST(HierarchicalLayoutCertification, EmptyAndSingletonLeavesAreCertified) {
  // Empty domains have no valid coordinate, so broadcast strides are harmless.
  const auto zero_extent =
      hl::make_layout(hl::make_tuple(0, 3), hl::make_tuple(0, 0));
  EXPECT_TRUE(zero_extent.empty());
  const auto disabled =
      hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(1, 1), false);
  EXPECT_TRUE(disabled.empty());
  // A singleton leaf contributes only offset zero, whatever its stride.
  const auto singleton =
      hl::make_layout(hl::make_tuple(hl::make_tuple(1, 4), 1),
                      hl::make_tuple(hl::make_tuple(0, 1), 1));
  EXPECT_EQ(singleton.offset(hl::make_tuple(3, 0)), 3);
  constexpr auto static_singleton = hl::make_layout(
      hl::make_tuple(hl::c<1>, hl::c<3>), hl::make_tuple(hl::c<0>, hl::c<1>));
  static_assert(static_singleton.required_span_size() == 3);
  constexpr auto rank_zero =
      hl::make_layout(hl::make_tuple(), hl::make_tuple());
  static_assert(rank_zero.offset(hl::make_tuple()) == 0);
}

TEST(HierarchicalLayoutCertification, CheckedArithmeticAtLimits) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  // The largest reachable offset is 1 + max / 2, which still fits.
  const auto widest =
      hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(1, maximum / 2));
  EXPECT_EQ(widest.required_span_size(), maximum / 2 + 2);
  EXPECT_EQ(widest.offset(hl::make_tuple(1, 1)), maximum / 2 + 1);
  // Unsorted leaves certify after sorting by stride.
  const auto descending =
      hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(maximum / 2, 1));
  EXPECT_EQ(descending.offset(hl::make_tuple(1, 1)), maximum / 2 + 1);
}

class HierarchicalLayoutCertificationOracle
    : public ::testing::TestWithParam<std::size_t> {};

// Every permutation of compact mixed-radix strides, optionally with a padding
// gap between leaves, is accepted. Each accepted map is then checked against
// an independent offset formula over its whole envelope.
TEST_P(HierarchicalLayoutCertificationOracle, ExhaustiveOracleOnAcceptedMaps) {
  const std::size_t gap = GetParam();
  const int permutations[6][3] = { { 0, 1, 2 }, { 0, 2, 1 }, { 1, 0, 2 },
                                   { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 } };
  for (std::size_t e0 = 0; e0 <= 3; ++e0) {
    for (std::size_t e1 = 0; e1 <= 3; ++e1) {
      for (std::size_t e2 = 0; e2 <= 3; ++e2) {
        const std::size_t extent[3] = { e0, e1, e2 };
        for (const auto &order : permutations) {
          // Leaf order[0] is fastest in storage, then order[1], then order[2].
          std::size_t stride[3]{};
          std::size_t next = 1;
          for (const int leaf : order) {
            stride[leaf] = next;
            next = (next * (extent[leaf] == 0 ? 1 : extent[leaf])) + gap;
          }
          const auto layout =
              hl::make_layout(hl::make_tuple(extent[0], extent[1], extent[2]),
                              hl::make_tuple(stride[0], stride[1], stride[2]));
          std::set<std::size_t> offsets;
          for (std::size_t i = 0; i < e0; ++i) {
            for (std::size_t j = 0; j < e1; ++j) {
              for (std::size_t k = 0; k < e2; ++k) {
                const auto coordinate = hl::make_tuple(i, j, k);
                const std::size_t expected =
                    i * stride[0] + j * stride[1] + k * stride[2];
                ASSERT_EQ(layout.offset(coordinate), expected);
                ASSERT_LT(expected, layout.required_span_size());
                ASSERT_TRUE(offsets.insert(expected).second);
              }
            }
          }
          ASSERT_EQ(offsets.size(), e0 * e1 * e2);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PaddingGaps, HierarchicalLayoutCertificationOracle,
                         ::testing::Values(0, 1, 5));

TEST(HierarchicalLayoutCertification, NestedMixedOracle) {
  // Depth three with static lanes and runtime blocks: ((4, (3, 2)), 5) with a
  // padded block stride, matching the hand-computed offsets below.
  const int blocks = 5;
  const auto layout = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<4>, hl::make_tuple(3, hl::c<2>)),
                     blocks),
      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::make_tuple(4, hl::c<16>)),
                     40));
  std::set<std::size_t> offsets;
  for (int b = 0; b < blocks; ++b) {
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
          const std::size_t expected = i + 4 * j + 16 * k + 40 * b;
          const auto nested =
              hl::make_tuple(hl::make_tuple(i, hl::make_tuple(j, k)), b);
          EXPECT_EQ(layout.offset(nested), expected);
          EXPECT_LT(expected, layout.required_span_size());
          EXPECT_TRUE(offsets.insert(expected).second);
        }
      }
    }
  }
}

TEST(HierarchicalLayoutCertification, StaticScratchFixtures) {
  using specfem::element::dimension_tag;
  using host = Kokkos::HostSpace;
  using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  // The production scratch types, placed in host memory over one buffer.
  using element2 = typename specfem::datatype::ScalarElementViewType<
      double, dimension_tag::dim2, 5, host, unmanaged>::type;
  using element3 = typename specfem::datatype::ScalarElementViewType<
      double, dimension_tag::dim3, 5, host, unmanaged>::type;
  // Chunk views use the execution space's array layout: LayoutRight on host
  // backends and LayoutLeft on CUDA/HIP. Both are instantiated explicitly.
  using chunk2_right = typename specfem::datatype::VectorChunkElementViewType<
      double, dimension_tag::dim2, 32, 5, 2, false, host, unmanaged,
      Kokkos::LayoutRight>::type;
  using chunk2_left = typename specfem::datatype::VectorChunkElementViewType<
      double, dimension_tag::dim2, 32, 5, 2, false, host, unmanaged,
      Kokkos::LayoutLeft>::type;
  using chunk3_left = typename specfem::datatype::VectorChunkElementViewType<
      double, dimension_tag::dim3, 64, 5, 3, false, host, unmanaged,
      Kokkos::LayoutLeft>::type;
  double buffer[64 * 5 * 5 * 5 * 3]{};
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      element2(buffer), cert::static_layout<true, 5, 5>()));
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      element3(buffer), cert::static_layout<true, 5, 5, 5>()));
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      chunk2_right(buffer), cert::static_layout<true, 32, 5, 5, 2>()));
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      chunk2_left(buffer), cert::static_layout<false, 32, 5, 5, 2>()));
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      chunk3_left(buffer), cert::static_layout<false, 64, 5, 5, 5, 3>()));
}

class HierarchicalLayoutCertificationDomainTile
    : public ::testing::TestWithParam<std::size_t> {};

// Compare with the production GPU mapping for CUDA (32) and HIP (64) storage
// tiles, at the ranks used by data_access/container.hpp.
TEST_P(HierarchicalLayoutCertificationDomainTile, MatchesDomainViewMapping) {
  const std::size_t offset_from_tile = GetParam();
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_domain_tile_family<32>(32 + offset_from_tile - 2));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_domain_tile_family<64>(64 + offset_from_tile - 2));
}

// Element counts of tile - 1, tile, and tile + 1.
INSTANTIATE_TEST_SUITE_P(TileBoundaries,
                         HierarchicalLayoutCertificationDomainTile,
                         ::testing::Values(1, 2, 3));

TEST(HierarchicalLayoutCertification, SingleElementOccupiesOnePaddedTile) {
  ASSERT_NO_FATAL_FAILURE(cert::expect_domain_tile_family<32>(1));
  ASSERT_NO_FATAL_FAILURE(cert::expect_domain_tile_family<64>(1));
}

template <typename ArrayLayout>
class HierarchicalLayoutCertificationNative : public ::testing::Test {};
using NativeLayouts = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;
TYPED_TEST_SUITE(HierarchicalLayoutCertificationNative, NativeLayouts);

// Contiguous views of ranks 1 to 6, including a singleton axis.
TYPED_TEST(HierarchicalLayoutCertificationNative, ContiguousRanksOneToSix) {
  using array_layout = TypeParam;
  using host = Kokkos::HostSpace;
  const Kokkos::View<double *, array_layout, host> rank1("rank1", 7);
  const Kokkos::View<double **, array_layout, host> rank2("rank2", 5, 3);
  const Kokkos::View<double ***, array_layout, host> rank3("rank3", 4, 1, 3);
  const Kokkos::View<double ****, array_layout, host> rank4("rank4", 3, 2, 1,
                                                            4);
  const Kokkos::View<double *****, array_layout, host> rank5("rank5", 2, 3, 2,
                                                             1, 3);
  const Kokkos::View<double ******, array_layout, host> rank6("rank6", 3, 2, 2,
                                                              1, 2, 3);
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank1, cert::runtime_layout(rank1)));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank2, cert::runtime_layout(rank2)));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank3, cert::runtime_layout(rank3)));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank4, cert::runtime_layout(rank4)));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank5, cert::runtime_layout(rank5)));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(rank6, cert::runtime_layout(rank6)));
  // A static extent and stride beside runtime values, on the unit-stride axis.
  constexpr std::size_t unit_axis =
      std::is_same_v<array_layout, Kokkos::LayoutLeft> ? 0 : 1;
  const auto mixed = [&] {
    if constexpr (unit_axis == 0) {
      return hl::make_layout(hl::make_tuple(hl::c<5>, rank2.extent(1)),
                             hl::make_tuple(hl::c<1>, rank2.stride(1)));
    } else {
      return hl::make_layout(hl::make_tuple(rank2.extent(0), hl::c<3>),
                             hl::make_tuple(rank2.stride(0), hl::c<1>));
    }
  }();
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(rank2, mixed));
}

// Subview expressions consumed by production kernels, in both array layouts.
TYPED_TEST(HierarchicalLayoutCertificationNative, ProductionSubviewFamilies) {
  using array_layout = TypeParam;
  using host = Kokkos::HostSpace;
  // S1: solver/forward_displacement_buffer.hpp, buf(slot, ALL, ALL).
  const Kokkos::View<double ***, array_layout, host> buffer("buffer", 3, 8, 5);
  const auto slot = Kokkos::subview(buffer, 1, Kokkos::ALL, Kokkos::ALL);
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(slot, cert::runtime_layout(slot)));
  // S2: compute_wavefield_helper.hpp, leading element range on rank 4 and 5.
  const Kokkos::View<double ****, array_layout, host> field2("field2", 10, 5, 5,
                                                             2);
  const auto chunk2 = Kokkos::subview(field2, Kokkos::make_pair(2, 7),
                                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(chunk2, cert::runtime_layout(chunk2)));
  const Kokkos::View<double *****, array_layout, host> field3("field3", 9, 3, 3,
                                                              3, 3);
  const auto chunk3 =
      Kokkos::subview(field3, Kokkos::make_pair(4, 9), Kokkos::ALL, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(chunk3, cert::runtime_layout(chunk3)));
  // S3: element_intersections.hpp, a fixed face and a face range.
  const Kokkos::View<int ***, array_layout, host> faces("faces", 6, 4, 4);
  const auto face = Kokkos::subview(faces, 2, Kokkos::ALL(), Kokkos::ALL());
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(face, cert::runtime_layout(face)));
  const auto face_range = Kokkos::subview(faces, Kokkos::make_pair(1, 4),
                                          Kokkos::ALL(), Kokkos::ALL());
  ASSERT_NO_FATAL_FAILURE(cert::expect_matches_native(
      face_range, cert::runtime_layout(face_range)));
  // S4: chunked_domain_iterator.hpp, a rank-1 index range.
  const Kokkos::View<int *, array_layout, host> indices("indices", 20);
  const auto range = Kokkos::subview(indices, Kokkos::make_pair(3, 11));
  ASSERT_NO_FATAL_FAILURE(
      cert::expect_matches_native(range, cert::runtime_layout(range)));
}

TEST(HierarchicalLayoutCertification, DeviceAcceptedFixtures) {
  const std::size_t elements = 33;
  Kokkos::View<int *> tiled("tiled", 2 * 800);
  Kokkos::View<int *> strided("strided", 3 * 8 * 5);
  Kokkos::deep_copy(tiled, -1);
  Kokkos::deep_copy(strided, -1);
  cert::write_certified_on_device(tiled.data(), strided.data(), elements);
  const auto tiled_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tiled);
  const auto strided_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, strided);
  using extents_type = Kokkos::dextents<std::size_t, 3>;
  const typename specfem::datatype::DomainViewMapping<
      specfem::element::dimension_tag::dim2, 32>::template mapping<extents_type>
      production(extents_type{ elements, 5, 5 });
  for (std::size_t i = 0; i < elements; ++i) {
    for (std::size_t z = 0; z < 5; ++z) {
      for (std::size_t x = 0; x < 5; ++x) {
        EXPECT_EQ(tiled_host(production(i, z, x)),
                  static_cast<int>(i * 25 + z * 5 + x));
      }
    }
  }
  const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      buffer(strided_host.data(), 3, 8, 5);
  for (int j = 0; j < 8; ++j) {
    for (int k = 0; k < 5; ++k) {
      EXPECT_EQ(buffer(1, j, k), j + 8 * k);
      EXPECT_EQ(buffer(0, j, k), -1);
    }
  }
}

namespace specfem::testing::layout_certification_impl {
class LayoutEnvironment : public ::testing::Environment {
public:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};
} // namespace specfem::testing::layout_certification_impl

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Death tests fork after Kokkos may have started host threads.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ::testing::AddGlobalTestEnvironment(
      new specfem::testing::layout_certification_impl::LayoutEnvironment);
  return RUN_ALL_TESTS();
}
