// Exercise debug contracts even when the surrounding preset defines NDEBUG.
// Overrides NDEBUG for this file only. Keep this target single-source and
// NO_UNITY, because other files would see a different layout.hpp (ODR).
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "specfem/experimental/layout/tile.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace hl = specfem::experimental::layout;

TEST(HierarchicalLayoutDebug, RejectsInvalidCoordinatesBeforeDecomposition) {
  const auto layout = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3)),
                                      hl::make_tuple(hl::make_tuple(1, 8)));
  EXPECT_FALSE(layout.valid(hl::make_tuple(12)));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            layout.offset(hl::make_tuple(12));
      },
      "coordinate out of bounds");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            layout.offset(hl::make_tuple(-1));
      },
      "coordinate out of bounds");
  const auto empty = hl::make_layout(hl::make_tuple(0), hl::make_tuple(1));
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            empty.offset(hl::make_tuple(0));
      },
      "coordinate out of bounds");
}

TEST(HierarchicalLayoutDebug, RejectsInvalidDescriptorsAndInsufficientBacking) {
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(-1), hl::make_tuple(1));
      },
      "nonnegative");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(1), hl::make_tuple(-1));
      },
      "nonnegative");
  const auto layout = hl::make_layout(hl::make_tuple(4), hl::make_tuple(2));
  int data[8]{};
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_view(data, 6, layout);
      },
      "backing capacity");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_view(static_cast<int *>(nullptr), 8, layout);
      },
      "nonnull backing");
  EXPECT_EQ(hl::make_view(data, 7, layout).layout().required_span_size(), 7);
}

TEST(HierarchicalLayoutDebug, RejectsOverflowButAcceptsEmptyEnvelope) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(maximum, 2), hl::make_tuple(1, 1));
      },
      "multiplication overflow");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(3), hl::make_tuple(maximum));
      },
      "multiplication overflow");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result =
            hl::make_layout(hl::make_tuple(2), hl::make_tuple(maximum));
      },
      "addition overflow");
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result = hl::make_layout(
            hl::make_tuple(2, 2), hl::make_tuple(maximum - 1, 2));
      },
      "addition overflow");
  bool enabled = true;
  EXPECT_DEATH(
      {
        [[maybe_unused]] const auto checked_result = hl::make_layout(
            hl::make_tuple(hl::c<2>), hl::make_tuple(maximum), enabled);
      },
      "addition overflow");
  const auto empty = hl::make_layout(hl::make_tuple(maximum, maximum, 0),
                                     hl::make_tuple(maximum, maximum, maximum));
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.required_span_size(), 0);
}

TEST(HierarchicalLayoutDebug, TransformCoordinates) {
  const auto parent = hl::make_layout(hl::make_tuple(2, hl::make_tuple(3, 2)),
                                      hl::make_tuple(1, hl::make_tuple(4, 16)));
  const auto p = hl::permute<hl::ModePath<1>, 1, 0>(parent);
  const auto g = hl::group<hl::ModePath<>, 0, 2>(p);
  EXPECT_EQ(g.offset(hl::make_tuple(11)), 25);
  EXPECT_FALSE(g.valid(hl::make_tuple(-1)));
  EXPECT_FALSE(g.valid(hl::make_tuple(12)));
  EXPECT_FALSE(p.valid(hl::make_tuple(0, hl::make_tuple(2, 0))));
  EXPECT_DEATH(static_cast<void>(g.offset(hl::make_tuple(-1))),
               "transformed coordinate");
  EXPECT_DEATH(static_cast<void>(g.offset(hl::make_tuple(12))),
               "transformed coordinate");
  EXPECT_DEATH(
      static_cast<void>(p.offset(hl::make_tuple(0, hl::make_tuple(2, 0)))),
      "transformed coordinate");
  const auto empty = hl::group<hl::ModePath<>, 0, 1>(
      hl::make_layout(hl::make_tuple(0), hl::make_tuple(1)));
  EXPECT_DEATH(static_cast<void>(empty.offset(hl::make_tuple(0))),
               "transformed coordinate");
}

TEST(HierarchicalLayoutDebug, SubviewSelections) {
  const auto m = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3), 2),
                                 hl::make_tuple(hl::make_tuple(1, 10), 5));
  EXPECT_DEATH(static_cast<void>(hl::subview(m, 12, hl::all)),
               "subview index out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, -1, hl::all)),
               "subview index out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::all, 2)),
               "subview index out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::range(5, 4), hl::all)),
               "range out of order or out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::range(0, 13), hl::all)),
               "range out of order or out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::range(-1, 4), hl::all)),
               "range out of order or out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::all, hl::range(1, 3))),
               "range out of order or out of range");
  // The range is checked before alignment, so no division sees a bad range.
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::range(1, 13), hl::all)),
               "range out of order or out of range");
  EXPECT_DEATH(static_cast<void>(hl::subview(m, hl::range(1, 5), hl::all)),
               "not aligned to nested mode blocks");
  EXPECT_DEATH(static_cast<void>(
                   hl::subview(m, hl::make_tuple(hl::all, hl::range(1, 4)), 1)),
               "range out of order or out of range");
  // Empty ranges need no alignment, including at the end of the mode.
  EXPECT_TRUE(hl::subview(m, hl::range(1, 1), hl::all).empty());
  EXPECT_TRUE(hl::subview(m, hl::range(12, 12), hl::all).empty());
  // A zero-extent mode has no index to fix.
  const auto zero = hl::make_layout(hl::make_tuple(0), hl::make_tuple(1));
  EXPECT_DEATH(static_cast<void>(hl::subview(zero, 0)),
               "subview index out of range");
}

TEST(HierarchicalLayoutDebug, SubviewCoordinates) {
  const auto m = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3), 2),
                                 hl::make_tuple(hl::make_tuple(1, 10), 5));
  const auto s = hl::subview(m, hl::range(4, 10), hl::all);
  EXPECT_EQ(s.offset(hl::make_tuple(5, 1)), 26);
  // Slot 6 is inside the rounded envelope but outside the selected range.
  EXPECT_FALSE(s.valid(hl::make_tuple(6, 0)));
  EXPECT_DEATH(static_cast<void>(s.offset(hl::make_tuple(6, 0))),
               "subview coordinate out of bounds");
  EXPECT_DEATH(static_cast<void>(s.offset(hl::make_tuple(8, 0))),
               "subview coordinate out of bounds");
  EXPECT_DEATH(
      static_cast<void>(s.offset(hl::make_tuple(hl::make_tuple(2, 1), 0))),
      "subview coordinate out of bounds");
}

TEST(HierarchicalLayoutDebug, DisplacementArithmetic) {
  // No valid subview can overflow: its displaced span never exceeds the
  // parent's. Exercise the checked arithmetic through the derive path.
  const auto maximum = std::numeric_limits<std::size_t>::max();
  using access = specfem::experimental::layout_impl::LayoutAccess;
  const auto largest =
      access::derive(hl::make_tuple(2), hl::make_tuple(1), true, maximum - 2);
  EXPECT_EQ(largest.required_span_size(), maximum);
  EXPECT_EQ(largest.offset(hl::make_tuple(1)), maximum - 1);
  EXPECT_DEATH(static_cast<void>(access::derive(
                   hl::make_tuple(2), hl::make_tuple(1), true, maximum - 1)),
               "addition overflow");
  EXPECT_DEATH(static_cast<void>(access::derive(
                   hl::make_tuple(), hl::make_tuple(), true, maximum)),
               "addition overflow");
  // A disabled descriptor needs no span, but any offset query is invalid.
  const auto disabled =
      access::derive(hl::make_tuple(2), hl::make_tuple(1), false, maximum);
  EXPECT_EQ(disabled.required_span_size(), 0);
  EXPECT_DEATH(static_cast<void>(disabled.offset(hl::make_tuple(1))),
               "coordinate out of bounds");
}

TEST(HierarchicalLayoutDebug, TileRequests) {
  const auto m = hl::make_layout(hl::make_tuple(7), hl::make_tuple(1));
  EXPECT_DEATH(static_cast<void>(hl::tile(m, 0)), "tile size must be positive");
  const auto nested = hl::make_layout(hl::make_tuple(hl::make_tuple(4, 3)),
                                      hl::make_tuple(hl::make_tuple(1, 4)));
  EXPECT_DEATH(static_cast<void>(hl::tile(nested, 3)),
               "must divide the fastest leaf");
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const auto large =
      hl::make_layout(hl::make_tuple(2), hl::make_tuple(maximum - 1));
  EXPECT_DEATH(static_cast<void>(hl::tile(large, 2)),
               "multiplication overflow");
  const auto huge = hl::make_layout(hl::make_tuple(maximum), hl::make_tuple(1));
  EXPECT_DEATH(static_cast<void>(hl::tile(huge, 2)), "multiplication overflow");
  EXPECT_DEATH(static_cast<void>(hl::tile(m, 4).offset(hl::make_tuple(7))),
               "tile coordinate out of bounds");
}
TEST(HierarchicalLayoutDebug, TileRegressionCertification) {
  EXPECT_DEATH(static_cast<void>(
                   hl::make_layout(hl::make_tuple(hl::make_tuple(4, 2), 3),
                                   hl::make_tuple(hl::make_tuple(1, 4), 7))),
               "uniqueness not certified");
}
