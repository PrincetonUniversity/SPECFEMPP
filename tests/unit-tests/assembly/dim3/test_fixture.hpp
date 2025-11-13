#pragma once

#include "MPI_environment.hpp"
#include "enumerations/interface.hpp"
#include "io/interface.hpp"
#include "mesh/dim3/meshfem3d/mesh.hpp"
#include "specfem/assembly.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct Assembly3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  specfem::assembly::assembly<dimension> assembly;

  Assembly3D() = default;

  Assembly3D(const std::string &database_file,
             const std::shared_ptr<specfem::MPI::MPI> mpi) {
    const auto mesh = specfem::io::meshfem3d::read_3d_mesh(database_file, *mpi);

    const int nspec = mesh.nspec;
    const int ngnod = mesh.control_nodes.ngnod;

    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    assembly.mesh = {
      nspec,     ngnod, 5, 5, 5, mesh.adjacency_graph, mesh.control_nodes,
      quadrature
    };
    assembly.jacobian_matrix = { assembly.mesh };
  }
};
} // namespace specfem::test_configuration

// Setup a fixture for parameterized tests
class Assembly3DTest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  specfem::test_configuration::Assembly3D assembly;

  Assembly3DTest() = default;

  void SetUp() override {
    const auto &folder = GetParam();
    const std::string database_file = "data/dim3/" + folder + "/database.bin";
    // Initialize MPI (assuming MPIEnvironment is defined elsewhere)
    auto mpi = MPIEnvironment::get_mpi();
    assembly = specfem::test_configuration::Assembly3D(database_file, mpi);
  }
  void TearDown() override {
    // Any cleanup needed for each test
  }

  ~Assembly3DTest() override = default;

  // Accessor for the assembly
  const auto &getAssembly() const { return assembly.assembly; }
};
