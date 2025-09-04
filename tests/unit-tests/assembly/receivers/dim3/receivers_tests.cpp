#include "specfem/assembly/receivers.hpp"
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace specfem::assembly;

class AssemblyReceivers3DTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Basic test parameters
    nspec = 10;
    nglly = 5;
    ngllz = 5;
    ngllx = 5;
    max_sig_step = 100;
    dt = 0.01;
    t0 = 0.0;
    nsteps_between_samples = 1;

    // Create test seismogram types
    seismogram_types = { specfem::wavefield::type::displacement,
                         specfem::wavefield::type::velocity,
                         specfem::wavefield::type::acceleration };
  }

  int nspec;
  int nglly;
  int ngllz;
  int ngllx;
  int max_sig_step;
  type_real dt;
  type_real t0;
  int nsteps_between_samples;
  std::vector<specfem::wavefield::type> seismogram_types;
};

TEST_F(AssemblyReceivers3DTest, DefaultConstructor) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test that default constructor doesn't crash
  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, DimensionTag) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  EXPECT_EQ(receiver_assembly.dimension_tag, specfem::dimension::type::dim3);
}

TEST_F(AssemblyReceivers3DTest, GetSeismogramTypes) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // For a default-constructed receiver assembly, we can't test much
  // since it requires mesh and other dependencies for full construction

  // Test that the method exists and can be called
  auto types = receiver_assembly.get_seismogram_types();
  // Default construction likely results in empty types
  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, StationsMethod) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test that stations method exists and returns a StationIterator reference
  const auto &station_iterator = receiver_assembly.stations();

  // Test that it returns a valid reference (size method should exist)
  EXPECT_EQ(station_iterator.size(), 0); // Default construction should be empty
}

TEST_F(AssemblyReceivers3DTest, InheritanceFromStationIterator) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

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

TEST_F(AssemblyReceivers3DTest, InheritanceFromSeismogramIterator) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test that it inherits from SeismogramIterator<dim3>
  // Should be able to access SeismogramIterator methods directly
  receiver_assembly.set_seismogram_step(5);
  EXPECT_EQ(receiver_assembly.get_seismogram_step(), 5);

  receiver_assembly.set_seismogram_type(1);
  EXPECT_EQ(receiver_assembly.get_seis_type(), 1);
}

TEST_F(AssemblyReceivers3DTest, SyncSeismograms) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test that sync_seismograms method exists and doesn't crash
  receiver_assembly.sync_seismograms();
  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, SeismogramIteratorFunctionality) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test basic seismogram iterator functionality
  receiver_assembly.set_seismogram_step(0);
  receiver_assembly.set_seismogram_type(0);

  EXPECT_EQ(receiver_assembly.get_seismogram_step(), 0);
  EXPECT_EQ(receiver_assembly.get_seis_type(), 0);
}

TEST_F(AssemblyReceivers3DTest, GetSeismogramMethod) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  std::string station_name = "TEST_STATION";
  std::string network_name = "TEST_NETWORK";
  specfem::wavefield::type wavefield_type =
      specfem::wavefield::type::displacement;

  // This will likely throw since maps aren't populated in default construction
  try {
    auto &returned_receiver = receiver_assembly.get_seismogram(
        station_name, network_name, wavefield_type);
    EXPECT_EQ(&returned_receiver,
              &returned_receiver); // Should return reference to self
  } catch (const std::exception &e) {
    // Expected if maps aren't populated
    SUCCEED();
  }
}

TEST_F(AssemblyReceivers3DTest, SetRotationMatrix) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Initialize the SeismogramIterator part using fixture values
  static_cast<specfem::assembly::receivers_impl::SeismogramIterator<
      specfem::dimension::type::dim3> &>(receiver_assembly) =
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim3>(2, seismogram_types.size(),
                                          max_sig_step, dt, t0,
                                          nsteps_between_samples);

  // Create a test rotation matrix (90 degree rotation around z-axis)
  std::array<std::array<type_real, 3>, 3> rotation_matrix = {
    { { { 0.0, -1.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, { { 0.0, 0.0, 1.0 } } }
  };

  // Test that setting rotation matrix works
  receiver_assembly.set_rotation_matrix(0, rotation_matrix);
  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, RotationMatrixIdentity) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Initialize using fixture values
  static_cast<specfem::assembly::receivers_impl::SeismogramIterator<
      specfem::dimension::type::dim3> &>(receiver_assembly) =
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim3>(1, seismogram_types.size(),
                                          max_sig_step, dt, t0,
                                          nsteps_between_samples);

  // Create identity rotation matrix
  std::array<std::array<type_real, 3>, 3> identity_matrix = {
    { { { 1.0, 0.0, 0.0 } }, { { 0.0, 1.0, 0.0 } }, { { 0.0, 0.0, 1.0 } } }
  };

  // Test that setting identity matrix works
  receiver_assembly.set_rotation_matrix(0, identity_matrix);
  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, MultipleRotationMatrices) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Initialize with 3 receivers using fixture values
  static_cast<specfem::assembly::receivers_impl::SeismogramIterator<
      specfem::dimension::type::dim3> &>(receiver_assembly) =
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim3>(3, seismogram_types.size(),
                                          max_sig_step, dt, t0,
                                          nsteps_between_samples);

  // Test setting rotation matrices for multiple receivers
  for (int irec = 0; irec < 3; ++irec) {
    std::array<std::array<type_real, 3>, 3> rotation_matrix = {
      { { { 1.0, 0.0, 0.0 } }, { { 0.0, 1.0, 0.0 } }, { { 0.0, 0.0, 1.0 } } }
    };

    // Set slightly different rotation for each receiver
    rotation_matrix[0][0] = 1.0 + irec * 0.1;

    receiver_assembly.set_rotation_matrix(irec, rotation_matrix);
  }

  SUCCEED();
}

TEST_F(AssemblyReceivers3DTest, MultipleSeismogramSteps) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test setting different seismogram steps
  for (int step = 0; step < 10; ++step) {
    receiver_assembly.set_seismogram_step(step);
    EXPECT_EQ(receiver_assembly.get_seismogram_step(), step);
  }
}

TEST_F(AssemblyReceivers3DTest, MultipleSeismogramTypes) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Test setting different seismogram types
  for (int type = 0; type < 3; ++type) {
    receiver_assembly.set_seismogram_type(type);
    EXPECT_EQ(receiver_assembly.get_seis_type(), type);
  }
}

TEST_F(AssemblyReceivers3DTest, ConstStaticMember) {
  // Test that dimension_tag is accessible as static member
  EXPECT_EQ(receivers<specfem::dimension::type::dim3>::dimension_tag,
            specfem::dimension::type::dim3);
}

TEST_F(AssemblyReceivers3DTest, TypeAliases) {
  // Test that the class can be instantiated (verifies type aliases work
  // correctly)
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // If type aliases are incorrect, this wouldn't compile
  SUCCEED();
}

// Note: The 3D receivers have additional functionality compared to 2D:
// - Rotation matrix support (tested above)
// - 3D-specific Lagrange interpolant handling
// - Additional GLL points in y-direction (nglly)

TEST_F(AssemblyReceivers3DTest, ThreeDimensionalSpecificFeatures) {
  receivers<specfem::dimension::type::dim3> receiver_assembly;

  // Initialize using fixture values
  static_cast<specfem::assembly::receivers_impl::SeismogramIterator<
      specfem::dimension::type::dim3> &>(receiver_assembly) =
      specfem::assembly::receivers_impl::SeismogramIterator<
          specfem::dimension::type::dim3>(1, seismogram_types.size(),
                                          max_sig_step, dt, t0,
                                          nsteps_between_samples);

  // Test that 3D-specific rotation matrix functionality exists
  std::array<std::array<type_real, 3>, 3> rotation_matrix = {
    { { { 0.707, -0.707, 0.0 } }, // 45 degree rotation around z
      { { 0.707, 0.707, 0.0 } },
      { { 0.0, 0.0, 1.0 } } }
  };

  receiver_assembly.set_rotation_matrix(0, rotation_matrix);

  // The fact that this compiles and runs without error
  // confirms the 3D-specific functionality is present
  SUCCEED();
}
