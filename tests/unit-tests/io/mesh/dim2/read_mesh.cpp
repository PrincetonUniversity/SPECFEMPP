#include "SPECFEM_Environment.hpp"

#include "read_mesh_test_fixture.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

using namespace specfem::test_configuration;

std::unordered_map<std::string, size_t> expected_2d_nelements = {
  { "HomogeneousMediumMPI4Procs", 4800 }
}; /// 80x60 mesh with 4800 elements total

TEST_P(Read2DMeshMPITest, SuccessfulExecution) {
  const auto &param_name = GetParam();

  // Get the mesh that was loaded in SetUp()
  const auto &mesh = getMesh();

  const auto expected_num_elements = expected_2d_nelements.at(param_name);

  // Synchronize all ranks after mesh loading
  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));

  // Get local element count and perform MPI reduction to sum across all ranks
  const auto local_num_elements = mesh.nspec; // Each rank's local element count
  int global_num_elements = 0;

  SPECFEM_MPI_SAFECALL(MPI_Reduce(&local_num_elements, &global_num_elements, 1,
                                  MPI_INT, MPI_SUM, 0,
                                  specfem::MPI::communicator()));

  // On rank 0, verify the mesh was loaded successfully
  SPECFEM_MPI_ON_ROOT({
    ASSERT_EQ(global_num_elements, expected_num_elements)
        << "Total number of elements across all ranks should match expected";

    specfem::Logger::info([&](std::ostringstream &oss) {
      oss << "io::read_2d_mesh test passed for dataset: " << param_name << "\n"
          << "Expected total elements: " << expected_num_elements << "\n"
          << "Total elements across " << specfem::MPI::get_size()
          << " processes: " << global_num_elements;
    });
  });

  // Final barrier to ensure all ranks complete the test
  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator()));
}
