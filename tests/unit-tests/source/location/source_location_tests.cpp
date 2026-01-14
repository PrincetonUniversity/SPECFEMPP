#include "../../SPECFEM_Environment.hpp"
#include "io/interface.hpp"
#include "medium/material.hpp"
#include "specfem/assembly.hpp"
#include "specfem/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <stdexcept>
#include <string>
#include <vector>

// ---- Read ground truth ---------

struct source_location {
  int islice, ispec;
  type_real xi, gamma;
};

void operator>>(YAML::Node &source_node, source_location &source_location) {
  source_location.xi = source_node["xi"].as<type_real>();
  source_location.gamma = source_node["gamma"].as<type_real>();
  source_location.islice = source_node["islice"].as<int>();
  source_location.ispec = source_node["ispec"].as<int>();
}

struct solution {
  int nnodes;
  std::vector<source_location> sources;
};

void operator>>(YAML::Node &node, solution &solution) {
  solution.nnodes = node["number-of-nodes"].as<int>();
  source_location source;
  YAML::Node source_locations = node["source-locations"];
  ASSERT_TRUE(source_locations.IsSequence());

  for (auto s : source_locations) {
    s >> source;
    solution.sources.push_back(source);
  }
}

std::vector<solution> parse_solution_file(std::string solution_file) {

  std::vector<solution> solutions;

  // parse solution file
  YAML::Node yaml = YAML::LoadFile(solution_file);
  int nsources = yaml["number-of-sources"].as<int>();
  int nsolutions = yaml["number-of-solutions"].as<int>();
  YAML::Node solutions_node = yaml["solution"];
  assert(solutions_node.IsSequence());
  solution solution;

  for (auto s : solutions_node) {
    s >> solution;
    solutions.push_back(solution);
  }

  assert(solutions.size() == nsolutions);

  return solutions;
}

// --------------------------------

// ------- Read test config -------

struct test_config {
  std::string sources_file, solutions_file, database_file;
};

void operator>>(YAML::Node &node, test_config &test_config) {
  test_config.sources_file = node["sources_file"].as<std::string>();
  test_config.solutions_file = node["solutions_file"].as<std::string>();
  test_config.database_file = node["database_file"].as<std::string>();
}

test_config parse_test_config(std::string config_filename) {
  YAML::Node yaml = YAML::LoadFile(config_filename);
  test_config test_config;

  yaml >> test_config;
  return test_config;
}
// ----------------------------

/**
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(SOURCES, compute_source_locations) {
  std::string config_filename = "source/test_config.yml";

  // parse solutions file for future use
  test_config test_config = parse_test_config(config_filename);
  std::vector<solution> solutions =
      parse_solution_file(test_config.solutions_file);

  // Set up GLL quadrature points
  specfem::quadrature::quadrature *gllx =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);
  specfem::quadrature::quadrature *gllz =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);

  // Read mesh for binary database for the test
  std::vector<std::shared_ptr<specfem::medium::material> > materials;
  specfem::mesh::mesh mesh =
      specfem::io::read_mesh(test_config.database_file, mpi);

  // read sources file
  auto [sources, t0] =
      specfem::io::read_sources(test_config.sources_file, 1.0, mpi);

  // setup compute struct for future use
  specfem::assembly::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                     gllz);
  specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
      jacobian_matrix(mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::assembly::properties material_properties(
      mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
      gllz->get_N());

  // Locate every source
  for (auto &source : sources)
    source->locate(compute.coordinates.coord, compute.h_ibool, gllx->get_hxi(),
                   gllz->get_hxi(), mesh.nproc, mesh.coorg,
                   mesh.material_ind.knods, mesh.npgeo,
                   material_properties.h_ispec_type, mpi);

  // flag to check if a solution exists for current MPI configuration
  bool tested = false;

  for (solution &solution : solutions) {
    if (specfem::MPI::get_size() == solution.nnodes) {
      tested = true;
      ASSERT_EQ(sources.size(), solution.sources.size());

      // check results for every source
      for (int i = 0; i < sources.size(); i++) {
        EXPECT_EQ(sources[i]->get_local_coordinates().ispec,
                  solution.sources[i].ispec - 1)
            << "For source " << i;
        EXPECT_EQ(sources[i]->get_local_coordinates().islice,
                  solution.sources[i].islice)
            << "For source " << i;
        EXPECT_NEAR(sources[i]->get_local_coordinates().xi,
                    solution.sources[i].xi, 1e-2)
            << "For source " << i;
        EXPECT_NEAR(sources[i]->get_local_coordinates().gamma,
                    solution.sources[i].gamma, 1e-2)
            << "For source " << i;
      }
    }
  }

  if (!tested)
    FAIL() << "Solution doesn't exist for current nnodes = "
           << specfem::MPI::get_size();
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
