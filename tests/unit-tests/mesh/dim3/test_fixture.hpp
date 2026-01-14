#pragma once

#include "SPECFEM_Environment.hpp"
#include "enumerations/interface.hpp"
#include "io/interface.hpp"
#include "specfem/mesh.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct ActualMesh3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  specfem::mesh::mesh<dimension> mesh;

  ActualMesh3D() = default;

  ActualMesh3D(const std::string &database_file) {
    mesh = specfem::io::read_3d_mesh(database_file);
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
    mesh = specfem::test_configuration::ActualMesh3D(database_file);
  }
  void TearDown() override {
    // Any cleanup needed for each test
  }

  ~Mesh3DTest() override = default;

  // Accessor for the mesh
  const auto &getMesh() const { return mesh.mesh; }
};
