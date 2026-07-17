#include "specfem/periodic_tasks/checkpointing.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {

constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
using checkpointing = specfem::periodic_tasks::checkpointing<dimension_tag>;
using periodic_task = specfem::periodic_tasks::periodic_task<dimension_tag>;

static_assert(std::is_base_of_v<periodic_task, checkpointing>);

TEST(Checkpointing, UsesPeriodicTaskCadence) {
  checkpointing task(4);

  EXPECT_TRUE(task.should_run(0));
  EXPECT_TRUE(task.should_run(8));
  EXPECT_FALSE(task.should_run(7));
  EXPECT_EQ(task.get_time_interval(), 4);
}

TEST(Checkpointing, CreatesFixedStrideReplayWindows) {
  const checkpointing task(4);

  EXPECT_EQ(task.replay_window(4, 10), std::make_pair(4, 8));
  EXPECT_EQ(task.replay_window(8, 10), std::make_pair(8, 10));
  EXPECT_EQ(task.max_window_size(), 4);
}

TEST(Checkpointing, RejectsInvalidInterval) {
  EXPECT_THROW(checkpointing(0), std::invalid_argument);
}

TEST(Checkpointing, CreatesSubdividedReplaySchedule) {
  const checkpointing task(256, 4);

  EXPECT_EQ(task.buffer_subdivisions(), 4);
  EXPECT_EQ(task.buffer_steps(), 64);
  EXPECT_EQ(task.checkpoint_slots(256), 3);
  EXPECT_EQ(task.split(256), 64);
  EXPECT_EQ(task.forward_steps(256), 448U);
}

TEST(Checkpointing, FallsBackToFullWindowReplay) {
  const checkpointing task(256, 1);

  EXPECT_EQ(task.buffer_subdivisions(), 1);
  EXPECT_EQ(task.buffer_steps(), 256);
  EXPECT_EQ(task.checkpoint_slots(256), 0);
  EXPECT_EQ(task.split(256), 0);
  EXPECT_EQ(task.forward_steps(256), 256U);
}

TEST(Checkpointing, PlacesShortSubdivisionFirst) {
  const checkpointing task(256, 5);

  EXPECT_EQ(task.buffer_steps(), 52);
  EXPECT_EQ(task.checkpoint_slots(256), 4);
  EXPECT_EQ(task.split(256), 48);
  EXPECT_EQ(task.forward_steps(256), 460U);
}

TEST(Checkpointing, RejectsInvalidSubdivisions) {
  EXPECT_THROW(checkpointing(4, 0), std::invalid_argument);
}

} // namespace
