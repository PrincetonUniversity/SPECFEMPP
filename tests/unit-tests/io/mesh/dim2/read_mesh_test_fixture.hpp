#pragma once

#include "SPECFEM_Environment.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct Read2DMesh {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim2;
  specfem::mesh::mesh<dimension> mesh;

  Read2DMesh() = default;

  Read2DMesh(const std::string &database) {
    mesh =
        specfem::io::read_2d_mesh(database, specfem::enums::elastic_wave::psv,
                                  specfem::enums::electromagnetic_wave::te,
                                  specfem::attenuation::Setup{});
  }
};
} // namespace specfem::test_configuration

// Setup a fixture for 2D parameterized tests
class Read2DMeshMPITest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim2;

  specfem::test_configuration::Read2DMesh mesh;

  Read2DMeshMPITest() = default;

  void SetUp() override {
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }

    if (specfem::MPI::communicator() == MPI_COMM_NULL) {
      GTEST_SKIP() << "Test designed for 4 processes. Rank "
                   << specfem::MPI::get_rank()
                   << " is outside the participating range [0-3].";
    }

    const auto &folder = GetParam();
    const std::string database = "data/mpi/dim2/" + folder + "/Database.bin";
    const auto mpi_database = specfem::MPI::format_proc_filename(database);
    mesh = specfem::test_configuration::Read2DMesh(mpi_database);
  }

  void TearDown() override {}

  ~Read2DMeshMPITest() override = default;

  const auto &getMesh() const { return mesh.mesh; }
};
