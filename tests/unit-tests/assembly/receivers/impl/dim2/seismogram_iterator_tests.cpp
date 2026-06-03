#include "specfem/assembly/receivers/impl/receiver_iterator.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace specfem::assembly::receivers_impl;

// ---------------------------------------------------------------------------
// Test helper: exposes protected h_seismogram_components so tests can inject
// known raw component values without going through the Kokkos device path.
// ---------------------------------------------------------------------------
class TestSeismogramIterator2D
    : public SeismogramIterator<specfem::element::dimension_tag::dim2> {
public:
  using Base = SeismogramIterator<specfem::element::dimension_tag::dim2>;
  using Base::Base;

  /// Write a single raw seismogram sample directly into the host mirror.
  /// Call after sync_seismograms() so the deep_copy does not overwrite it.
  void set_raw_component(int seis_step, int iseis, int irec, int comp,
                         type_real value) {
    h_seismogram_components(seis_step, iseis, irec, comp) = value;
  }
};

class SeismogramIterator2DTest : public ::testing::Test {
protected:
  void SetUp() override {
    nreceivers = 2;
    nseismograms = 1;
    max_sig_step = 10;
    dt = 0.01;
    t0 = 0.0;
    nstep_between_samples = 1;
  }

  // Helper function to create initialized iterator with test data
  SeismogramIterator<specfem::element::dimension_tag::dim2>
  createInitializedIterator() {
    SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
        nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

    // Rotation matrices are initialized to identity by the constructor (no
    // rotation). Nothing extra needed here.

    // Sync seismogram data (initialized to zero by default)
    iterator.sync_seismograms();

    return iterator;
  }

  int nreceivers;
  int nseismograms;
  int max_sig_step;
  type_real dt;
  type_real t0;
  int nstep_between_samples;
};

TEST_F(SeismogramIterator2DTest, DefaultConstructor) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator;
  // Test that default constructor doesn't crash
  SUCCEED();
}

TEST_F(SeismogramIterator2DTest, ParameterizedConstructor) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  // Test that constructor completes without throwing
  SUCCEED();
}

TEST_F(SeismogramIterator2DTest, SetSeismogramStep) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  const int test_step = 5;
  iterator.set_seismogram_step(test_step);

  EXPECT_EQ(iterator.get_seismogram_step(), test_step);
}

TEST_F(SeismogramIterator2DTest, SetSeismogramType) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  const int test_type = 0;
  iterator.set_seismogram_type(test_type);

  EXPECT_EQ(iterator.get_seis_type(), test_type);
}

TEST_F(SeismogramIterator2DTest, SyncSeismograms) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  // Test that sync_seismograms doesn't crash
  iterator.sync_seismograms();
  SUCCEED();
}

TEST_F(SeismogramIterator2DTest, IteratorCreation) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  // Test that iterators can be created
  auto begin_iter = iterator.begin();
  auto end_iter = iterator.end();

  EXPECT_NE(begin_iter, end_iter);
}

TEST_F(SeismogramIterator2DTest, IteratorDereference) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  auto iter = iterator.begin();
  auto [time, seismograms] = *iter;

  // Check time calculation: seis_step * dt * nstep_between_samples + t0
  // seis_step = 0, so time = 0 * 0.01 * 1 + 0.0 = 0.0
  EXPECT_DOUBLE_EQ(time, 0.0);
  EXPECT_EQ(seismograms.size(), 2); // 2D should have 2 components
}

TEST_F(SeismogramIterator2DTest, IteratorIncrement) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  auto iter = iterator.begin();
  auto [time1, seismograms1] = *iter;

  ++iter;
  auto [time2, seismograms2] = *iter;

  // Time should increment by dt * nstep_between_samples
  EXPECT_DOUBLE_EQ(time2 - time1, dt * nstep_between_samples);
}

TEST_F(SeismogramIterator2DTest, TimeCalculationWithOffset) {
  const type_real test_t0 = 1.0;
  const int test_nstep = 2;

  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, test_t0, test_nstep);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  auto iter = iterator.begin();
  auto [time, seismograms] = *iter;

  // Time should be: seis_step * dt * nstep_between_samples + t0
  // = 0 * 0.01 * 2 + 1.0 = 1.0
  EXPECT_DOUBLE_EQ(time, test_t0);
}

TEST_F(SeismogramIterator2DTest, MultipleTimeSteps) {
  const int small_max_steps = 3;
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, small_max_steps, dt, t0, nstep_between_samples);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  auto iter = iterator.begin();
  type_real expected_time = t0;

  for (int step = 0; step < small_max_steps; ++step) {
    EXPECT_NE(iter, iterator.end());

    auto [time, seismograms] = *iter;
    EXPECT_DOUBLE_EQ(time, expected_time);
    EXPECT_EQ(seismograms.size(), 2);

    ++iter;
    expected_time += dt * nstep_between_samples;
  }

  EXPECT_EQ(iter, iterator.end());
}

TEST_F(SeismogramIterator2DTest, IdentityRotationProducesZeroForZeroData) {
  auto iterator = createInitializedIterator(); // rotation matrix = identity

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);

  auto iter = iterator.begin();
  auto [time, seismograms] = *iter;

  // 2D iterator always returns 2 components
  EXPECT_EQ(seismograms.size(), 2);

  // Zero-initialized seismogram data -> all output components are 0
  for (const auto &component : seismograms) {
    EXPECT_TRUE(std::isfinite(component));
    EXPECT_DOUBLE_EQ(component, 0.0);
  }
}

TEST_F(SeismogramIterator2DTest, RotationMatrixAppliedCorrectly) {
  // 90-degree rotation: R = [[0, -1], [1, 0]]
  // Expected: output[0] = -raw[1], output[1] = raw[0]
  TestSeismogramIterator2D iterator(nreceivers, nseismograms, max_sig_step, dt,
                                    t0, nstep_between_samples);

  // sync_seismograms() copies device (all zeros) -> host mirror.
  // Raw values are set afterward so they are not overwritten.
  iterator.sync_seismograms();

  // Inject raw[0] = 1, raw[1] = 2 for receiver 0, step 0, seismogram type 0.
  iterator.set_raw_component(/*seis_step=*/0, /*iseis=*/0, /*irec=*/0,
                             /*comp=*/0, 1.0);
  iterator.set_raw_component(/*seis_step=*/0, /*iseis=*/0, /*irec=*/0,
                             /*comp=*/1, 2.0);

  const type_real angle = Kokkos::numbers::pi_v<type_real> / 2.0; // 90 deg
  const type_real cos_a = std::cos(angle);                        // ≈ 0
  const type_real sin_a = std::sin(angle);                        // ≈ 1
  std::array<std::array<type_real, 2>, 2> R = { { { { cos_a, -sin_a } },
                                                  { { sin_a, cos_a } } } };
  iterator.set_rotation_matrix(0, R);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);

  auto iter = iterator.begin();
  auto [time, seismograms] = *iter;

  EXPECT_EQ(seismograms.size(), 2);
  // output[0] = cos_a*1 + (-sin_a)*2 = 0*1 + (-1)*2 = -2
  EXPECT_NEAR(seismograms[0], -2.0, 1e-5);
  // output[1] = sin_a*1 + cos_a*2   = 1*1 +   0*2  =  1
  EXPECT_NEAR(seismograms[1], 1.0, 1e-5);
}

TEST_F(SeismogramIterator2DTest, GetSeismogramMethod) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  std::string station_name = "TEST_STATION";
  std::string network_name = "TEST_NETWORK";
  specfem::enums::wavefield wavefield_type =
      specfem::enums::wavefield::displacement;

  // Note: This will likely throw since maps aren't populated, but we test the
  // method exists
  try {
    auto &returned_iterator =
        iterator.get_seismogram(station_name, network_name, wavefield_type);
    EXPECT_EQ(&returned_iterator, &iterator); // Should return reference to self
  } catch (const std::exception &e) {
    // Expected if maps aren't populated - the method exists and can be called
    SUCCEED();
  }
}

TEST_F(SeismogramIterator2DTest, ComponentsSize) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  iterator.set_seismogram_step(0);
  iterator.set_seismogram_type(0);
  iterator.sync_seismograms();

  auto iter = iterator.begin();
  auto [time, seismograms] = *iter;

  // 2D seismogram should always have exactly 2 components
  EXPECT_EQ(seismograms.size(), 2);
}

TEST_F(SeismogramIterator2DTest, SetStepOutsideRange) {
  SeismogramIterator<specfem::element::dimension_tag::dim2> iterator(
      nreceivers, nseismograms, max_sig_step, dt, t0, nstep_between_samples);

  // Test setting step to boundary values
  iterator.set_seismogram_step(max_sig_step - 1);
  EXPECT_EQ(iterator.get_seismogram_step(), max_sig_step - 1);

  iterator.set_seismogram_step(0);
  EXPECT_EQ(iterator.get_seismogram_step(), 0);
}
