#include "specfem/periodic_tasks/wavefield_checkpoint.hpp"
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {

constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;
using wavefield_checkpoint =
    specfem::periodic_tasks::wavefield_checkpoint<dimension_tag>;
using periodic_task = specfem::periodic_tasks::periodic_task<dimension_tag>;

static_assert(std::is_base_of_v<periodic_task, wavefield_checkpoint>);

TEST(WavefieldCheckpoint, UsesPeriodicTaskCadence) {
  wavefield_checkpoint task(4);

  EXPECT_TRUE(task.should_run(0));
  EXPECT_TRUE(task.should_run(8));
  EXPECT_FALSE(task.should_run(7));
  EXPECT_EQ(task.get_time_interval(), 4);
}

TEST(WavefieldCheckpoint, CreatesFixedStrideReplayWindows) {
  const wavefield_checkpoint task(4);

  EXPECT_EQ(task.replay_window(4, 10), std::make_pair(4, 8));
  EXPECT_EQ(task.replay_window(8, 10), std::make_pair(8, 10));
  EXPECT_EQ(task.max_window_size(), 4);
}

TEST(WavefieldCheckpoint, RejectsInvalidInterval) {
  EXPECT_THROW(wavefield_checkpoint(0), std::invalid_argument);
}

TEST(WavefieldCheckpoint, CreatesSubdividedReplaySchedule) {
  const wavefield_checkpoint task(256, 4);

  EXPECT_EQ(task.buffer_subdivisions(), 4);
  EXPECT_EQ(task.buffer_steps(), 64);
  EXPECT_EQ(task.checkpoint_slots(256), 3);
  EXPECT_EQ(task.split(256), 64);
  EXPECT_EQ(task.forward_steps(256), 448U);
}

TEST(WavefieldCheckpoint, FallsBackToFullWindowReplay) {
  const wavefield_checkpoint task(256, 1);

  EXPECT_EQ(task.buffer_subdivisions(), 1);
  EXPECT_EQ(task.buffer_steps(), 256);
  EXPECT_EQ(task.checkpoint_slots(256), 0);
  EXPECT_EQ(task.split(256), 0);
  EXPECT_EQ(task.forward_steps(256), 256U);
}

TEST(WavefieldCheckpoint, PlacesShortSubdivisionFirst) {
  const wavefield_checkpoint task(256, 5);

  EXPECT_EQ(task.buffer_steps(), 52);
  EXPECT_EQ(task.checkpoint_slots(256), 4);
  EXPECT_EQ(task.split(256), 48);
  EXPECT_EQ(task.forward_steps(256), 460U);
}

TEST(WavefieldCheckpoint, RejectsInvalidSubdivisions) {
  EXPECT_THROW(wavefield_checkpoint(4, 0), std::invalid_argument);
}

} // namespace
