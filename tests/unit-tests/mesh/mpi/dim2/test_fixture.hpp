#pragma once

#include "SPECFEM_Environment.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct ActualMesh2D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim2;
  specfem::mesh::mesh<dimension> mesh;

  ActualMesh2D() = default;

  ActualMesh2D(const std::string &database) {
    mesh = specfem::io::read_2d_mesh(
        database, specfem::enums::elastic_wave::psv,
        specfem::enums::electromagnetic_wave::te, false);
  }
};
} // namespace specfem::test_configuration

// Setup a fixture for parameterized tests
class MPIMesh2DTest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim2;

  specfem::test_configuration::ActualMesh2D mesh;

  MPIMesh2DTest() = default;

  void SetUp() override {
    // Check if MPI size requirement was met during environment setup
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }

    // Check if the current rank is within the participating range for this test
    if (specfem::MPI::communicator() == MPI_COMM_NULL) {
      GTEST_SKIP() << "Test designed for 4 processes. Rank "
                   << specfem::MPI::get_rank()
                   << " is outside the participating range [0-3].";
    }
    const auto &folder = GetParam();
    const std::string database = "data/mpi/dim2/" + folder + "/Database.bin";
    const auto mpi_database = specfem::MPI::format_proc_filename(database);
    mesh = specfem::test_configuration::ActualMesh2D(mpi_database);
  }
  void TearDown() override {
    // Any cleanup needed for each test
  }

  ~MPIMesh2DTest() override = default;

  // Accessor for the mesh
  const auto &getMesh() const { return mesh.mesh; }
};
