#pragma once

#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include <gtest/gtest.h>
#include <string>

namespace specfem::test_configuration {
struct AssemblyMPI3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;
  specfem::assembly::mesh<dimension> assembly_mesh; ///< Assembly mesh with GLL
                                                    ///< point coordinates and
                                                    ///< index mapping
  specfem::assembly::mpi<dimension> mpi_interfaces; ///< MPI communication
                                                    ///< groups for face data
                                                    ///< exchange between
                                                    ///< partitions

  AssemblyMPI3D() = default;

  AssemblyMPI3D(const specfem::mesh::mesh<dimension> &mesh);
};
} // namespace specfem::test_configuration

// Setup a fixture for parameterized assembly MPI tests
class AssemblyMPI3DTest : public ::testing::TestWithParam<std::string> {
protected:
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;
  specfem::test_configuration::AssemblyMPI3D assembly;
  specfem::mesh::mesh<dimension> mesh;

  AssemblyMPI3DTest() = default;

  void SetUp() override;

  void TearDown() override;

  ~AssemblyMPI3DTest() override = default;

  // Accessor for the assembly
  const auto &getMPIInterfaces() const { return assembly.mpi_interfaces; }
  const auto &getMesh() const { return mesh; }
  const auto &getAssemblyMesh() const { return assembly.assembly_mesh; }
};
