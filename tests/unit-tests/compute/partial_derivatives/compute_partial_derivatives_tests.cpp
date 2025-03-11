#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "IO/interface.hpp"
#include "compute/interface.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename, xix_file, xiz_file, gammax_file, gammaz_file,
      jacobian_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.xix_file = Node["xix_file"].as<std::string>();
  test_config.xiz_file = Node["xiz_file"].as<std::string>();
  test_config.gammax_file = Node["gammax_file"].as<std::string>();
  test_config.gammaz_file = Node["gammaz_file"].as<std::string>();
  test_config.jacobian_file = Node["jacobian_file"].as<std::string>();

  return;
}

test_config get_test_config(std::string config_filename,
                            specfem::MPI::MPI *mpi) {
  // read test config file
  YAML::Node yaml = YAML::LoadFile(config_filename);
  test_config test_config{};
  if (mpi->get_size() == 1) {
    YAML::Node Node = yaml["SerialTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == 1);
    for (auto N : database)
      N >> test_config;
  } else {
    YAML::Node Node = yaml["ParallelTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == Node["config"]["nproc"].as<int>());
    assert(mpi->get_size() == Node["config"]["nproc"].as<int>());
    for (auto N : database) {
      if (N["processor"].as<int>() == mpi->get_rank())
        N >> test_config;
    }
  }

  return test_config;
}
// ---------------------------------------------------------------------------

/**
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(COMPUTE_TESTS, compute_partial_derivatives) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename = "compute/partial_derivatives/test_config.yml";
  test_config test_config = get_test_config(config_filename, mpi);

  // Set up GLL quadrature points
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  specfem::mesh::mesh mesh =
      specfem::IO::read_2d_mesh(test_config.database_filename, mpi);

  specfem::compute::mesh compute_mesh(mesh.tags, mesh.control_nodes,
                                      quadratures);
  specfem::compute::partial_derivatives partial_derivatives(compute_mesh);

  const int nspec = compute_mesh.control_nodes.nspec;
  const int ngllz = compute_mesh.quadratures.gll.N;
  const int ngllx = compute_mesh.quadratures.gll.N;

  specfem::testing::array3d<double, Kokkos::LayoutRight> xix_ref(
      test_config.xix_file, nspec, ngllz, ngllx);
  specfem::testing::array3d<double, Kokkos::LayoutRight> gammax_ref(
      test_config.gammax_file, nspec, ngllz, ngllx);
  specfem::testing::array3d<double, Kokkos::LayoutRight> gammaz_ref(
      test_config.gammaz_file, nspec, ngllz, ngllx);
  specfem::testing::array3d<double, Kokkos::LayoutRight> jacobian_ref(
      test_config.jacobian_file, nspec, ngllz, ngllx);

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ispec = 0; ispec < nspec; ++ispec) {
        const specfem::point::index<specfem::dimension::type::dim2> index(
            ispec, iz, ix);
        const auto point_partial_derivatives = [&]() {
          specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, false>
              point_partial_derivatives;
          specfem::compute::load_on_host(index, partial_derivatives,
                                         point_partial_derivatives);
          return point_partial_derivatives;
        }();
        const int ispec_mesh = compute_mesh.mapping.compute_to_mesh(ispec);

        EXPECT_NEAR(point_partial_derivatives.xix,
                    xix_ref.data(ispec_mesh, iz, ix), xix_ref.tol);
        EXPECT_NEAR(point_partial_derivatives.gammax,
                    gammax_ref.data(ispec_mesh, iz, ix), gammax_ref.tol);
        EXPECT_NEAR(point_partial_derivatives.gammaz,
                    gammaz_ref.data(ispec_mesh, iz, ix), gammaz_ref.tol);
        EXPECT_NEAR(point_partial_derivatives.jacobian,
                    jacobian_ref.data(ispec_mesh, iz, ix), jacobian_ref.tol);
      }
    }
  }

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      constexpr static int vector_length =
          specfem::datatype::simd<type_real, true>::size();

      for (int ispec = 0; ispec < nspec; ispec += vector_length) {
        const int num_elements =
            (ispec + vector_length < nspec) ? vector_length : nspec - ispec;
        const specfem::point::simd_index<specfem::dimension::type::dim2>
            simd_index(ispec, num_elements, iz, ix);
        const auto point_partial_derivatives = [&]() {
          specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                              true, true>
              point_partial_derivatives;
          specfem::compute::load_on_host(simd_index, partial_derivatives,
                                         point_partial_derivatives);
          return point_partial_derivatives;
        }();

        for (int i = 0; i < num_elements; ++i) {
          const int ispec_mesh =
              compute_mesh.mapping.compute_to_mesh(ispec + i);
          EXPECT_NEAR(point_partial_derivatives.xix[i],
                      xix_ref.data(ispec_mesh, iz, ix), xix_ref.tol);
          EXPECT_NEAR(point_partial_derivatives.gammax[i],
                      gammax_ref.data(ispec_mesh, iz, ix), gammax_ref.tol);
          EXPECT_NEAR(point_partial_derivatives.gammaz[i],
                      gammaz_ref.data(ispec_mesh, iz, ix), gammaz_ref.tol);
          EXPECT_NEAR(point_partial_derivatives.jacobian[i],
                      jacobian_ref.data(ispec_mesh, iz, ix), jacobian_ref.tol);
        }
      }
    }
  }

  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
