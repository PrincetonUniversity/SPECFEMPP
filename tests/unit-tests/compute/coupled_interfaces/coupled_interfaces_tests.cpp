#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "IO/interface.hpp"
#include "compute/interface.hpp"
#include "edge/interface.hpp"
#include "mesh/mesh.hpp"
#include "point/coordinates.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Reading test config

namespace test_config {
struct mesh {
public:
  std::string database_filename;
};

struct elastic_acoustic {
  std::string elastic_ispec_file;
  std::string acoustic_ispec_file;
};

struct elastic_poroelastic {
  std::string elastic_ispec_file;
  std::string poroelastic_ispec_file;
};

struct acoustic_poroelastic {
  std::string acoustic_ispec_file;
  std::string poroelastic_ispec_file;
};

struct databases {
public:
  test_config::mesh mesh;
  test_config::elastic_acoustic elastic_acoustic;
  test_config::elastic_poroelastic elastic_poroelastic;
  test_config::acoustic_poroelastic acoustic_poroelastic;
};

struct configuration {
public:
  int processors;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node config = Node["config"];
    configuration.processors = config["nproc"].as<int>();
    YAML::Node database = Node["databases"];

    assert(database.IsMap());

    YAML::Node elastic_acoustic = database["elastic_acoustic"];
    YAML::Node elastic_poroelastic = database["elastic_poroelastic"];
    YAML::Node acoustic_poroelastic = database["acoustic_poroelastic"];
    YAML::Node mesh = database["mesh"];

    assert(elastic_acoustic.IsMap());
    assert(elastic_poroelastic.IsMap());
    assert(acoustic_poroelastic.IsMap());

    databases.mesh.database_filename = mesh["database"].as<std::string>();

    databases.elastic_acoustic.elastic_ispec_file =
        elastic_acoustic["elastic_ispec"].as<std::string>();
    databases.elastic_acoustic.acoustic_ispec_file =
        elastic_acoustic["acoustic_ispec"].as<std::string>();

    databases.elastic_poroelastic.elastic_ispec_file =
        elastic_poroelastic["elastic_ispec"].as<std::string>();
    databases.elastic_poroelastic.poroelastic_ispec_file =
        elastic_poroelastic["poroelastic_ispec"].as<std::string>();

    databases.acoustic_poroelastic.acoustic_ispec_file =
        acoustic_poroelastic["acoustic_ispec"].as<std::string>();
    databases.acoustic_poroelastic.poroelastic_ispec_file =
        acoustic_poroelastic["poroelastic_ispec"].as<std::string>();

    return;
  }
  std::string name;
  std::string description;
  test_config::configuration configuration;
  test_config::databases databases;
};
} // namespace test_config
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_config::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_config::Test(N));

  return;
}

// ---------------------------------------------------------------------------

void test_edges(
    const specfem::kokkos::HostView4d<type_real> coordinates,
    const specfem::kokkos::HostMirror1d<int> ispec1,
    const specfem::kokkos::HostMirror1d<int> ispec2,
    const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge1,
    const specfem::kokkos::HostMirror1d<specfem::edge::interface> edge2) {

  const int num_interfaces = ispec1.extent(0);
  const int ngllx = coordinates.extent(3);
  const int ngllz = coordinates.extent(2);

  for (int interface = 0; interface < num_interfaces; interface++) {
    const int ispec1l = ispec1(interface);
    const int ispec2l = ispec2(interface);

    const auto edge1l = edge1(interface);
    const auto edge2l = edge2(interface);

    // iterate over the edge
    int npoints = specfem::edge::num_points_on_interface(edge1l);

    for (int ipoint = 0; ipoint < npoints; ipoint++) {
      // Get ipoint along the edge in element1
      int i1, j1;
      specfem::edge::locate_point_on_self_edge(ipoint, edge1l, j1, i1);
      const specfem::point::gcoord2 self_point(coordinates(0, ispec1l, j1, i1),
                                               coordinates(1, ispec1l, j1, i1));

      // Get ipoint along the edge in element2
      int i2, j2;
      specfem::edge::locate_point_on_coupled_edge(ipoint, edge2l, j2, i2);
      const specfem::point::gcoord2 coupled_point(
          coordinates(0, ispec2l, j2, i2), coordinates(1, ispec2l, j2, i2));

      const type_real distance =
          specfem::point::distance(self_point, coupled_point);

      // Check that the distance between the two points is small

      ASSERT_TRUE(distance < 1.e-10)
          << "Invalid edges found at interface number " << interface;
    }
  }
}

TEST(COMPUTE_TESTS, coupled_interfaces_tests) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename = "compute/coupled_interfaces/test_config.yaml";

  std::vector<test_config::Test> tests;
  parse_test_config(YAML::LoadFile(config_filename), tests);

  // Set up GLL quadrature points
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  for (auto Test : tests) {
    std::cout << "Executing test: " << Test.name << std::endl;

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh =
        specfem::IO::read_2d_mesh(Test.databases.mesh.database_filename, mpi);

    // Generate compute structs to be used by the solver
    specfem::compute::mesh assembly(mesh.control_nodes, quadratures);

    specfem::compute::properties properties(assembly.nspec, assembly.ngllz,
                                            assembly.ngllx, mesh.materials);

    try {
      // Generate coupled interfaces struct to be used by the solver
      specfem::compute::coupled_interfaces coupled_interfaces(
          assembly, properties, mesh.coupled_interfaces);

      // Test coupled interfaces
      // Check if the mesh was read correctly
      specfem::kokkos::HostView1d<int> h_elastic_ispec;
      specfem::kokkos::HostView1d<int> h_acoustic_ispec;
      specfem::kokkos::HostView1d<int> h_poroelastic_ispec;

      if (Test.databases.elastic_acoustic.elastic_ispec_file != "NULL") {
        h_elastic_ispec =
            coupled_interfaces.elastic_acoustic.h_medium1_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight> elastic_ispec_array(
            h_elastic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight> elastic_ispec_ref(
            Test.databases.elastic_acoustic.elastic_ispec_file,
            coupled_interfaces.elastic_acoustic.num_interfaces);

        ASSERT_TRUE(elastic_ispec_array == elastic_ispec_ref);

        h_acoustic_ispec =
            coupled_interfaces.elastic_acoustic.h_medium2_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight>
            acoustic_ispec_array(h_acoustic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight> acoustic_ispec_ref(
            Test.databases.elastic_acoustic.acoustic_ispec_file,
            coupled_interfaces.elastic_acoustic.num_interfaces);

        ASSERT_TRUE(acoustic_ispec_array == acoustic_ispec_ref);

        // test if the edges match
        test_edges(assembly.points.h_coord,
                   coupled_interfaces.elastic_acoustic.h_medium1_index_mapping,
                   coupled_interfaces.elastic_acoustic.h_medium2_index_mapping,
                   coupled_interfaces.elastic_acoustic.h_medium1_edge_type,
                   coupled_interfaces.elastic_acoustic.h_medium2_edge_type);
      } else {
        ASSERT_TRUE(coupled_interfaces.elastic_acoustic.num_interfaces == 0);
      }
      // Check if the mesh was read correctly
      if (Test.databases.elastic_poroelastic.elastic_ispec_file != "NULL") {
        h_elastic_ispec =
            coupled_interfaces.elastic_poroelastic.h_medium1_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight> elastic_ispec_array(
            h_elastic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight> elastic_ispec_ref(
            Test.databases.elastic_poroelastic.elastic_ispec_file,
            coupled_interfaces.elastic_poroelastic.num_interfaces);

        ASSERT_TRUE(elastic_ispec_array == elastic_ispec_ref);

        h_poroelastic_ispec =
            coupled_interfaces.elastic_poroelastic.h_medium2_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight>
            poroelastic_ispec_array(h_poroelastic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight>
            poroelastic_ispec_ref(
                Test.databases.elastic_poroelastic.poroelastic_ispec_file,
                coupled_interfaces.elastic_poroelastic.num_interfaces);

        ASSERT_TRUE(poroelastic_ispec_array == poroelastic_ispec_ref);

        // test if the edges match
        test_edges(
            assembly.points.h_coord,
            coupled_interfaces.elastic_poroelastic.h_medium1_index_mapping,
            coupled_interfaces.elastic_poroelastic.h_medium2_index_mapping,
            coupled_interfaces.elastic_poroelastic.h_medium1_edge_type,
            coupled_interfaces.elastic_poroelastic.h_medium2_edge_type);
      } else {
        ASSERT_TRUE(coupled_interfaces.elastic_poroelastic.num_interfaces == 0);
      }

      // Check if the mesh was read correctly
      if (Test.databases.acoustic_poroelastic.acoustic_ispec_file != "NULL") {
        h_poroelastic_ispec =
            coupled_interfaces.acoustic_poroelastic.h_medium2_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight>
            poroelastic_ispec_array(h_poroelastic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight>
            poroelastic_ispec_ref(
                Test.databases.acoustic_poroelastic.poroelastic_ispec_file,
                coupled_interfaces.acoustic_poroelastic.num_interfaces);

        ASSERT_TRUE(poroelastic_ispec_array == poroelastic_ispec_ref);

        h_acoustic_ispec =
            coupled_interfaces.acoustic_poroelastic.h_medium1_index_mapping;
        specfem::testing::array1d<int, Kokkos::LayoutRight>
            acoustic_ispec_array(h_acoustic_ispec);

        specfem::testing::array1d<int, Kokkos::LayoutRight> acoustic_ispec_ref(
            Test.databases.acoustic_poroelastic.acoustic_ispec_file,
            coupled_interfaces.acoustic_poroelastic.num_interfaces);

        ASSERT_TRUE(acoustic_ispec_array == acoustic_ispec_ref);

        // test if the edges match
        test_edges(
            assembly.points.h_coord,
            coupled_interfaces.acoustic_poroelastic.h_medium1_index_mapping,
            coupled_interfaces.acoustic_poroelastic.h_medium2_index_mapping,
            coupled_interfaces.acoustic_poroelastic.h_medium1_edge_type,
            coupled_interfaces.acoustic_poroelastic.h_medium2_edge_type);
      } else {
        ASSERT_TRUE(coupled_interfaces.acoustic_poroelastic.num_interfaces ==
                    0);
      }

      std::cout << " - Test passed\n";
    } catch (std::exception &e) {
      std::cout << " - Error: " << e.what() << std::endl;
      FAIL() << " Test failed\n"
             << " - Test name: " << Test.name << "\n"
             << " - Number of MPI processors: " << Test.configuration.processors
             << "\n"
             << " - Error: " << e.what() << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
