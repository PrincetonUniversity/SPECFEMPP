#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "compute/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
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
  std::string database_filename, rho_file, kappa_file, mu_file, rho_vs_file,
      rho_vp_file, qkappa_file, qmu_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.rho_file = Node["rho_file"].as<std::string>();
  test_config.kappa_file = Node["kappa_file"].as<std::string>();
  test_config.mu_file = Node["mu_file"].as<std::string>();
  test_config.rho_vs_file = Node["rho_vs_file"].as<std::string>();
  test_config.rho_vp_file = Node["rho_vp_file"].as<std::string>();
  test_config.qkappa_file = Node["qkappa_file"].as<std::string>();
  test_config.qmu_file = Node["qmu_file"].as<std::string>();

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
TEST(COMPUTE_TESTS, compute_elastic_properties) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename =
      "../../../tests/unit-tests/compute/elastic/test_config.yml";
  test_config test_config = get_test_config(config_filename, mpi);

  // Set up GLL quadrature points
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  specfem::mesh::mesh mesh(test_config.database_filename, mpi);

  std::cout << mesh.print() << std::endl;

  const int nspec = mesh.nspec;
  const int ngllz = gll.get_N();
  const int ngllx = gll.get_N();

  specfem::compute::mesh compute_mesh(mesh.tags, mesh.control_nodes,
                                      quadratures);
  specfem::compute::properties compute_properties(
      nspec, ngllz, ngllx, compute_mesh.mapping, mesh.tags, mesh.materials);

  specfem::testing::array3d<type_real, Kokkos::LayoutRight> rho_global(
      test_config.rho_file, nspec, ngllz, ngllx);
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> mu_global(
      test_config.mu_file, nspec, ngllz, ngllx);
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> kappa_global(
      test_config.kappa_file, nspec, ngllz, ngllx);

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ispec = 0; ispec < nspec; ++ispec) {
        specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz,
                                                                    ix);
        if (compute_properties.h_element_types(ispec) ==
                specfem::element::medium_tag::elastic &&
            compute_properties.h_element_property(ispec) ==
                specfem::element::property_tag::isotropic) {
          const auto properties =
              [&]() -> specfem::point::properties<
                        specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false> {
            specfem::point::properties<
                specfem::dimension::type::dim2,
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic, false>
                properties;
            specfem::compute::load_on_host(index, compute_properties,
                                           properties);
            return properties;
          }();
          const int ispec_mesh = compute_mesh.mapping.compute_to_mesh(ispec);
          const auto kappa = properties.lambdaplus2mu - properties.mu;
          EXPECT_FLOAT_EQ(properties.rho, rho_global.data(ispec_mesh, iz, ix));
          EXPECT_FLOAT_EQ(properties.mu, mu_global.data(ispec_mesh, iz, ix));
          EXPECT_FLOAT_EQ(kappa, kappa_global.data(ispec_mesh, iz, ix));
        }
      }
    }
  }

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      constexpr int vector_length =
          specfem::datatype::simd<type_real, true>::size();

      for (int ispec = 0; ispec < nspec; ispec += vector_length) {
        const int num_elements =
            (ispec + vector_length < nspec) ? vector_length : nspec - ispec;
        const specfem::point::simd_index<specfem::dimension::type::dim2>
            simd_index(ispec, num_elements, iz, ix);
        if (compute_properties.h_element_types(ispec) ==
                specfem::element::medium_tag::elastic &&
            compute_properties.h_element_property(ispec) ==
                specfem::element::property_tag::isotropic) {
          const auto properties =
              [&]() -> specfem::point::properties<
                        specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, true> {
            specfem::point::properties<
                specfem::dimension::type::dim2,
                specfem::element::medium_tag::elastic,
                specfem::element::property_tag::isotropic, true>
                properties;
            specfem::compute::load_on_host(simd_index, compute_properties,
                                           properties);
            return properties;
          }();
          const auto kappa = properties.lambdaplus2mu - properties.mu;
          for (int i = 0; i < num_elements; ++i) {
            const int ispec_mesh =
                compute_mesh.mapping.compute_to_mesh(ispec + i);
            EXPECT_FLOAT_EQ(properties.rho[i],
                            rho_global.data(ispec_mesh, iz, ix));
            EXPECT_FLOAT_EQ(properties.mu[i],
                            mu_global.data(ispec_mesh, iz, ix));
            EXPECT_FLOAT_EQ(kappa[i], kappa_global.data(ispec_mesh, iz, ix));
          }
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
