#include "specfem/assembly/receivers.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace specfem::assembly;

class AssemblyReceivers2DTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Basic test parameters
    nspec = 10;
    ngllz = 5;
    ngllx = 5;
    max_sig_step = 100;
    dt = 0.01;
    t0 = 0.0;
    nsteps_between_samples = 1;

    // Create test seismogram types
    seismogram_types = { specfem::wavefield::type::displacement,
                         specfem::wavefield::type::velocity };
  }

  int nspec;
  int ngllz;
  int ngllx;
  int max_sig_step;
  type_real dt;
  type_real t0;
  int nsteps_between_samples;
  std::vector<specfem::wavefield::type> seismogram_types;
};

TEST_F(AssemblyReceivers2DTest, DefaultConstructor) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test that default constructor doesn't crash
  SUCCEED();
}

TEST_F(AssemblyReceivers2DTest, DimensionTag) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  EXPECT_EQ(receiver_assembly.dimension_tag, specfem::dimension::type::dim2);
}

TEST_F(AssemblyReceivers2DTest, GetSeismogramTypes) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // For a default-constructed receiver assembly, we can't test much
  // since it requires mesh and other dependencies for full construction

  // Test that the method exists and can be called
  auto types = receiver_assembly.get_seismogram_types();
  // Default construction likely results in empty types
  SUCCEED();
}

TEST_F(AssemblyReceivers2DTest, StationsMethod) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test that stations method exists and returns a StationIterator reference
  const auto &station_iterator = receiver_assembly.stations();

  // Test that it returns a valid reference (size method should exist)
  EXPECT_EQ(station_iterator.size(), 0); // Default construction should be empty
}

TEST_F(AssemblyReceivers2DTest, InheritanceFromStationIterator) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test that it inherits from StationIterator
  // Should be able to access StationIterator methods directly
  EXPECT_EQ(receiver_assembly.size(), 0); // From StationIterator

  // Use explicit cast to avoid ambiguity between StationIterator and
  // SeismogramIterator
  const auto &station_iter =
      static_cast<const specfem::assembly::receivers_impl::StationIterator &>(
          receiver_assembly);
  EXPECT_EQ(station_iter.begin(), station_iter.end()); // From StationIterator
}

TEST_F(AssemblyReceivers2DTest, InheritanceFromSeismogramIterator) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test that it inherits from SeismogramIterator<dim2>
  // Should be able to access SeismogramIterator methods directly
  receiver_assembly.set_seismogram_step(5);
  EXPECT_EQ(receiver_assembly.get_seismogram_step(), 5);

  receiver_assembly.set_seismogram_type(1);
  EXPECT_EQ(receiver_assembly.get_seis_type(), 1);
}

TEST_F(AssemblyReceivers2DTest, SyncSeismograms) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test that sync_seismograms method exists and doesn't crash
  receiver_assembly.sync_seismograms();
  SUCCEED();
}

TEST_F(AssemblyReceivers2DTest, SeismogramIteratorFunctionality) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test basic seismogram iterator functionality
  receiver_assembly.set_seismogram_step(0);
  receiver_assembly.set_seismogram_type(0);

  EXPECT_EQ(receiver_assembly.get_seismogram_step(), 0);
  EXPECT_EQ(receiver_assembly.get_seis_type(), 0);
}

TEST_F(AssemblyReceivers2DTest, GetSeismogramMethod) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  std::string station_name = "TEST_STATION";
  std::string network_name = "TEST_NETWORK";
  specfem::wavefield::type wavefield_type =
      specfem::wavefield::type::displacement;

  // This will likely throw since maps aren't populated in default construction
  try {
    auto &returned_receiver = receiver_assembly.get_seismogram(
        station_name, network_name, wavefield_type);
    EXPECT_EQ(&returned_receiver,
              &receiver_assembly); // Should return reference to self
  } catch (const std::exception &e) {
    // Expected if maps aren't populated
    SUCCEED();
  }
}

// Note: Full constructor tests would require creating mock mesh, tags, and
// element_types objects which would need significant infrastructure. These
// tests focus on the basic functionality that can be tested without complex
// dependencies.

TEST_F(AssemblyReceivers2DTest, MultipleSeismogramSteps) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test setting different seismogram steps
  for (int step = 0; step < 10; ++step) {
    receiver_assembly.set_seismogram_step(step);
    EXPECT_EQ(receiver_assembly.get_seismogram_step(), step);
  }
}

TEST_F(AssemblyReceivers2DTest, MultipleSeismogramTypes) {
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // Test setting different seismogram types
  for (int type = 0; type < 3; ++type) {
    receiver_assembly.set_seismogram_type(type);
    EXPECT_EQ(receiver_assembly.get_seis_type(), type);
  }
}

TEST_F(AssemblyReceivers2DTest, ConstStaticMember) {
  // Test that dimension_tag is accessible as static member
  EXPECT_EQ(receivers<specfem::dimension::type::dim2>::dimension_tag,
            specfem::dimension::type::dim2);
}

TEST_F(AssemblyReceivers2DTest, TypeAliases) {
  // Test that the class can be instantiated (verifies type aliases work
  // correctly)
  receivers<specfem::dimension::type::dim2> receiver_assembly;

  // If type aliases are incorrect, this wouldn't compile
  SUCCEED();
}
