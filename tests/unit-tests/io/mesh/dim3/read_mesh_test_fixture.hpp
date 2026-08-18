#pragma once

#include "SPECFEM_Environment.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct Read3DMesh {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;
  specfem::mesh::mesh<dimension> mesh;

  Read3DMesh() = default;

  Read3DMesh(const std::string &database) {
    mesh = specfem::io::read_3d_mesh(database, specfem::attenuation::Setup{});
  }
};
} // namespace specfem::test_configuration

// Setup a fixture for parameterized tests
class Read3DMeshMPITest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  specfem::test_configuration::Read3DMesh mesh;

  Read3DMeshMPITest() = default;

  void SetUp() override {
    // Check if MPI size requirement was met during environment setup
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }

    // Check if the current
    if (specfem::MPI::communicator() == MPI_COMM_NULL) {
      GTEST_SKIP() << "Test designed for 4 processes. Rank "
                   << specfem::MPI::get_rank()
                   << " is outside the participating range [0-3].";
    }

    const auto &folder = GetParam();
    const std::string database = "data/mpi/dim3/" + folder + "/Database.bin";
    const auto mpi_database = specfem::MPI::format_proc_filename(database);
    mesh = specfem::test_configuration::Read3DMesh(mpi_database);
  }

  void TearDown() override {
    // Any cleanup needed for each test
  }

  ~Read3DMeshMPITest() override = default;

  // Accessor for the mesh
  const auto &getMesh() const { return mesh.mesh; }
};
