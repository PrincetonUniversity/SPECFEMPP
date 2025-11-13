#pragma once

#include "MPI_environment.hpp"
#include "enumerations/interface.hpp"
#include "io/interface.hpp"
#include "mesh/dim3/meshfem3d/mesh.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct ActualMesh3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  specfem::mesh::meshfem3d::mesh<dimension> mesh;

  ActualMesh3D() = default;

  ActualMesh3D(const std::string &database_file,
               std::shared_ptr<specfem::MPI::MPI> mpi) {
    mesh = specfem::io::meshfem3d::read_3d_mesh(database_file, *mpi);
  }
};
} // namespace specfem::test_configuration

// Setup a fixture for parameterized tests
class Mesh3DTest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  specfem::test_configuration::ActualMesh3D mesh;

  Mesh3DTest() = default;

  void SetUp() override {
    const auto &folder = GetParam();
    const std::string database_file = "data/dim3/" + folder + "/database.bin";
    // Get MPI from test environment (returns shared_ptr)
    auto mpi = MPIEnvironment::get_mpi();
    mesh = specfem::test_configuration::ActualMesh3D(database_file, mpi);
  }
  void TearDown() override {
    // Any cleanup needed for each test
  }

  ~Mesh3DTest() override = default;

  // Accessor for the mesh
  const auto &getMesh() const { return mesh.mesh; }
};
