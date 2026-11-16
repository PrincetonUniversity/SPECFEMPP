#include "specfem/experimental/layout/layout.hpp"
#include "specfem/experimental/layout/view.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <set>
#include <type_traits>
#include <utility>

namespace hl = specfem::experimental::layout;

namespace specfem::testing::hierarchical_layout_impl {
template <typename Node>
KOKKOS_INLINE_FUNCTION constexpr std::size_t metadata_total(const Node &node);

template <typename Node, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
metadata_children_total(const Node &node, std::index_sequence<I...>);

void write_device_storage(int *pointer);
} // namespace specfem::testing::hierarchical_layout_impl

template <typename Node, std::size_t... I>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
specfem::testing::hierarchical_layout_impl::metadata_children_total(
    const Node &node, std::index_sequence<I...>) {
  return (std::size_t{ 0 } + ... + metadata_total(hl::get<I>(node)));
}

template <typename Node>
KOKKOS_INLINE_FUNCTION constexpr std::size_t
specfem::testing::hierarchical_layout_impl::metadata_total(const Node &node) {
  if constexpr (hl::is_tuple_v<Node>) {
    return metadata_children_total(
        node, std::make_index_sequence<hl::tuple_size_v<Node>>{});
  } else {
    return static_cast<std::size_t>(node);
  }
}

void specfem::testing::hierarchical_layout_impl::write_device_storage(
    int *pointer) {
  const auto captured_mapping = hl::make_layout(
      hl::make_tuple(hl::make_tuple(2, hl::make_tuple(3, 2)), 2),
      hl::make_tuple(hl::make_tuple(1, hl::make_tuple(4, 16)), 32));
  const auto captured_view = hl::make_view(pointer, 58, captured_mapping);
  Kokkos::parallel_for(
      "hierarchical_layout_indexing", 24, KOKKOS_LAMBDA(int i) {
        const auto mapping = hl::make_layout(
            hl::make_tuple(
                hl::make_tuple(hl::c<2>, hl::make_tuple(3, hl::c<2>)),
                hl::c<2>),
            hl::make_tuple(
                hl::make_tuple(hl::c<1>, hl::make_tuple(hl::c<4>, 16)),
                hl::c<32>));
        static_assert(hl::tuple_size_v<decltype(mapping.shape())> == 2);
        const auto view = hl::make_view(pointer, 58, mapping);
        const auto scalar = hl::make_tuple(i % 12, i / 12);
        const auto nested = hl::make_tuple(
            hl::make_tuple(i % 2, hl::make_tuple((i / 2) % 3, (i / 6) % 2)),
            i / 12);
        const auto empty =
            hl::make_layout(hl::make_tuple(0), hl::make_tuple(1));
        const auto rank_zero =
            hl::make_layout(hl::make_tuple(), hl::make_tuple());
        const bool correct = mapping.valid(nested) &&
                             !mapping.valid(hl::make_tuple(12, 0)) &&
                             mapping.offset(scalar) == mapping.offset(nested) &&
                             empty.empty() && empty.required_span_size() == 0 &&
                             rank_zero.valid(hl::make_tuple()) &&
                             rank_zero.offset(hl::make_tuple()) == 0;
        view(scalar) = correct ? i : -2;
        if (captured_view(nested) != i) {
          view(scalar) = -3;
        }
      });
}

TEST(HierarchicalLayout, MixedRecursiveMetadata) {
  const auto shape = hl::make_tuple(
      hl::make_tuple(hl::c<4>, hl::make_tuple(hl::c<3>, hl::c<2>)), 5);
  const auto strides = hl::make_tuple(
      hl::make_tuple(hl::c<1>, hl::make_tuple(hl::c<4>, hl::c<12>)), 24);
  const auto layout = hl::make_layout(shape, strides);
  static_assert(decltype(layout)::rank() == 2);
  static_assert(decltype(layout.extent<0>())::value == 24);
  EXPECT_EQ(layout.extent<1>(), 5);
  EXPECT_EQ(layout.envelope_size(), 120);
  EXPECT_EQ(layout.required_span_size(), 120);
  static_assert(std::remove_cvref_t<decltype(hl::get<0>(
                    hl::get<0>(layout.shape())))>::value == 4);
}

TEST(HierarchicalLayout, PublicTraitsSupportRecursiveMetadataTraversal) {
  const auto shape = hl::make_tuple(hl::make_tuple(hl::c<4>, 3), hl::c<2>);
  const auto mapping = hl::make_layout(
      shape, hl::make_tuple(hl::make_tuple(hl::c<1>, 16), hl::c<4>));
  static_assert(hl::is_tuple_v<decltype(mapping.shape())>);
  static_assert(hl::tuple_size_v<decltype(mapping.shape())> == 2);
  static_assert(hl::tuple_size_v<decltype(hl::get<0>(shape))> == 2);
  static_assert(!hl::is_tuple_v<decltype(hl::get<1>(shape))>);
  static_assert(!hl::is_tuple_v<int>);
  static_assert(hl::tuple_size_v<decltype(hl::make_tuple())> == 0);
  EXPECT_EQ(specfem::testing::hierarchical_layout_impl::metadata_total(
                mapping.shape()),
            9);
  constexpr auto empty = hl::make_tuple();
  static_assert(
      specfem::testing::hierarchical_layout_impl::metadata_total(empty) == 0);
}

class HierarchicalLayoutOracle : public ::testing::TestWithParam<int> {};

TEST_P(HierarchicalLayoutOracle, ScalarAndNestedCoordinatesMatchStorageOracle) {
  const int outer = GetParam();
  const auto layout =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<4>, outer), hl::c<2>),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 16), hl::c<4>));
  std::set<std::size_t> offsets;
  for (int block = 0; block < outer; ++block) {
    for (int lane = 0; lane < 4; ++lane) {
      for (int component = 0; component < 2; ++component) {
        const auto nested =
            hl::make_tuple(hl::make_tuple(lane, block), component);
        const auto scalar = hl::make_tuple(block * 4 + lane, component);
        const std::size_t expected = block * 16 + component * 4 + lane;
        EXPECT_TRUE(layout.valid(nested));
        EXPECT_EQ(layout.offset(nested), expected);
        EXPECT_EQ(layout.offset(scalar), expected);
        EXPECT_LT(expected, layout.required_span_size());
        EXPECT_TRUE(offsets.insert(layout.offset(scalar)).second);
      }
    }
  }
  EXPECT_EQ(offsets.size(), outer * 8);
  EXPECT_FALSE(layout.valid(hl::make_tuple(outer * 4, 0)));
  EXPECT_FALSE(layout.valid(hl::make_tuple(-1, 0)));
  EXPECT_FALSE(layout.valid(hl::make_tuple(0, 2)));
  EXPECT_FALSE(layout.valid(hl::make_tuple(hl::make_tuple(4, 0), 0)));
  EXPECT_EQ(layout.required_span_size(), outer == 0 ? 0 : (outer - 1) * 16 + 8);
}

INSTANTIATE_TEST_SUITE_P(SmallShapes, HierarchicalLayoutOracle,
                         ::testing::Values(0, 1, 2, 3, 7));

TEST(HierarchicalLayout, BorrowedStoragePreservesConstnessAndCapacity) {
  int storage[48]{};
  const auto layout =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<4>, 3), hl::c<2>),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 16), hl::c<4>));
  const auto view = hl::make_view(storage, 48, layout);
  view(hl::make_tuple(5, 1)) = 73;
  EXPECT_EQ(storage[21], 73);
  const auto readonly =
      hl::make_view(static_cast<const int *>(storage), 48, layout);
  static_assert(
      std::is_same_v<decltype(readonly(hl::make_tuple(5, 1))), const int &>);
  EXPECT_EQ(readonly(hl::make_tuple(5, 1)), 73);
}

TEST(HierarchicalLayout, IntegralLeavesAndWideCoordinateBounds) {
  const auto layout =
      hl::make_layout(hl::make_tuple(hl::c<char{ 4 }>, char{ 2 }),
                      hl::make_tuple(char{ 1 }, hl::c<4>));
  EXPECT_EQ(layout.offset(hl::make_tuple(char{ 3 }, short{ 1 })), 7);
  EXPECT_FALSE(layout.valid(
      hl::make_tuple(std::numeric_limits<unsigned long long>::max(), 0)));
  EXPECT_FALSE(layout.valid(hl::make_tuple(static_cast<signed char>(-1), 0)));
}

TEST(HierarchicalLayout, EmptyDomainsAndRankZeroAvoidArithmetic) {
  constexpr auto scalar = hl::make_layout(hl::make_tuple(), hl::make_tuple());
  static_assert(scalar.rank() == 0);
  static_assert(scalar.valid(hl::make_tuple()));
  static_assert(scalar.offset(hl::make_tuple()) == 0);
  static_assert(scalar.envelope_size() == 1);
  static_assert(scalar.required_span_size() == 1);
  constexpr auto disabled =
      hl::make_layout(hl::make_tuple(), hl::make_tuple(), false);
  static_assert(disabled.empty());
  static_assert(!disabled.valid(hl::make_tuple()));
  static_assert(disabled.required_span_size() == 0);
  const auto empty_view =
      hl::make_view(static_cast<int *>(nullptr), 0, disabled);
  EXPECT_TRUE(empty_view.layout().empty());
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  constexpr auto empty =
      hl::make_layout(hl::make_tuple(hl::c<maximum>, hl::c<maximum>, hl::c<0>),
                      hl::make_tuple(hl::c<maximum>, hl::c<maximum>, hl::c<1>));
  static_assert(empty.envelope_size() == 0);
  static_assert(empty.required_span_size() == 0);
  EXPECT_FALSE(empty.valid(hl::make_tuple(0, 0, 0)));
  constexpr auto singleton =
      hl::make_layout(hl::make_tuple(hl::c<1>), hl::make_tuple(hl::c<maximum>));
  static_assert(singleton.required_span_size() == 1);
  constexpr auto largest =
      hl::make_layout(hl::make_tuple(hl::c<maximum>), hl::make_tuple(hl::c<1>));
  static_assert(largest.required_span_size() == maximum);
  static_assert(largest.offset(hl::make_tuple(hl::c<maximum - 1>)) ==
                maximum - 1);
}

TEST(HierarchicalLayout, DisabledDomainsIgnoreUnreachableStaticBackingSpan) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  constexpr auto static_mapping = hl::make_layout(
      hl::make_tuple(hl::c<2>), hl::make_tuple(hl::c<maximum>), false);
  constexpr auto mixed_mapping =
      hl::make_layout(hl::make_tuple(hl::c<2>), hl::make_tuple(maximum), false);
  constexpr auto runtime_mapping =
      hl::make_layout(hl::make_tuple(2), hl::make_tuple(maximum), false);
  static_assert(static_mapping.empty());
  static_assert(static_mapping.required_span_size() == 0);
  static_assert(mixed_mapping.required_span_size() == 0);
  static_assert(runtime_mapping.required_span_size() == 0);
  static_assert(static_mapping.envelope_size() == 2);
  bool enabled = false;
  const auto mapping = hl::make_layout(hl::make_tuple(hl::c<2>),
                                       hl::make_tuple(hl::c<maximum>), enabled);
  EXPECT_FALSE(mapping.valid(hl::make_tuple(0)));
  EXPECT_EQ(mapping.required_span_size(), 0);
  const auto view = hl::make_view(static_cast<int *>(nullptr), 0, mapping);
  EXPECT_TRUE(view.layout().empty());
}

TEST(HierarchicalLayout, DeepScalarSubtreesAndDeviceBorrowedWrites) {
  Kokkos::View<int *> data("layout_storage", 58);
  Kokkos::deep_copy(data, -1);
  const auto pointer = data.data();
  specfem::testing::hierarchical_layout_impl::write_device_storage(pointer);
  const auto result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);
  std::set<int> visited;
  for (int d = 0; d < 2; ++d) {
    for (int c = 0; c < 2; ++c) {
      for (int b = 0; b < 3; ++b) {
        for (int a = 0; a < 2; ++a) {
          const int offset = a + 4 * b + 16 * c + 32 * d;
          EXPECT_EQ(result(offset), a + 2 * b + 6 * c + 12 * d);
          visited.insert(offset);
        }
      }
    }
  }
  for (int i = 0; i < 58; ++i) {
    if (!visited.contains(i)) {
      EXPECT_EQ(result(i), -1);
    }
  }
}

namespace specfem::testing::hierarchical_layout_impl {
class LayoutEnvironment : public ::testing::Environment {
public:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};
} // namespace specfem::testing::hierarchical_layout_impl

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new specfem::testing::hierarchical_layout_impl::LayoutEnvironment);
  return RUN_ALL_TESTS();
}
