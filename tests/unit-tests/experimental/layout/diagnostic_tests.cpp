// Exercise debug contracts even when the surrounding preset defines NDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "specfem/experimental/layout/view.hpp"
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
            hl::make_tuple(hl::c<2>),
            hl::make_tuple(hl::c<std::numeric_limits<std::size_t>::max()>),
            enabled);
      },
      "addition overflow");
  const auto empty = hl::make_layout(hl::make_tuple(maximum, maximum, 0),
                                     hl::make_tuple(maximum, maximum, maximum));
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.required_span_size(), 0);
}
