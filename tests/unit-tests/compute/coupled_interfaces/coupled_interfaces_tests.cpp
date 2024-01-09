#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/compare_array.h"
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
    const specfem::kokkos::HostMirror3d<int> h_ibool,
    const specfem::kokkos::HostView2d<type_real> coord,
    const specfem::kokkos::HostMirror1d<int> ispec1,
    const specfem::kokkos::HostMirror1d<int> ispec2,
    const specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge1,
    const specfem::kokkos::HostMirror1d<specfem::enums::edge::type> edge2) {

  const int num_interfaces = ispec1.extent(0);
  const int ngllx = h_ibool.extent(2);
  const int ngllz = h_ibool.extent(1);

  for (int interface = 0; interface < num_interfaces; interface++) {
    const int ispec1l = ispec1(interface);
    const int ispec2l = ispec2(interface);

    const auto edge1l = edge1(interface);
    const auto edge2l = edge2(interface);

    // iterate over the edge
    int npoints = specfem::compute::coupled_interfaces::access::npoints(
        edge1l, ngllx, ngllz);

    for (int ipoint = 0; ipoint < npoints; ipoint++) {
      // Get ipoint along the edge in element1
      int i1, j1;
      specfem::compute::coupled_interfaces::access::coupled_iterator(
          ipoint, edge1l, ngllx, ngllz, i1, j1);
      const int iglob1 = h_ibool(ispec1l, j1, i1);

      // Get ipoint along the edge in element2
      int i2, j2;
      specfem::compute::coupled_interfaces::access::self_iterator(
          ipoint, edge2l, ngllx, ngllz, i2, j2);
      const int iglob2 = h_ibool(ispec2l, j2, i2);

      // Check that the distance between the two points is small

      ASSERT_TRUE((((coord(0, iglob1) - coord(0, iglob2)) *
                    (coord(0, iglob1) - coord(0, iglob2))) +
                   ((coord(1, iglob1) - coord(1, iglob2)) *
                    (coord(1, iglob1) - coord(1, iglob2)))) < 1.e-10)
          << "Invalid edges found at interface number " << interface;
    }
  }
}

TEST(COMPUTE_TESTS, coupled_interfaces_tests) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename =
      "../../../tests/unit-tests/compute/coupled_interfaces/test_config.yaml";

  std::vector<test_config::Test> tests;
  parse_test_config(YAML::LoadFile(config_filename), tests);

  // Set up GLL quadrature points
  specfem::quadrature::quadrature *gllx =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);
  specfem::quadrature::quadrature *gllz =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);
  std::vector<std::shared_ptr<specfem::material::material> > materials;

  for (auto Test : tests) {
    std::cout << "Executing test: " << Test.name << std::endl;

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh(Test.databases.mesh.database_filename, materials,
                             mpi);

    // Generate compute structs to be used by the solver
    specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                      gllz);

    // Generate properties struct to be used by the solver
    specfem::compute::properties properties(mesh.material_ind.kmato, materials,
                                            mesh.nspec, gllz->get_N(),
                                            gllx->get_N());

    try {
      // Generate coupled interfaces struct to be used by the solver
      specfem::compute::coupled_interfaces::coupled_interfaces
          coupled_interfaces(compute.h_ibool, compute.coordinates.coord,
                             properties.h_ispec_type, mesh.coupled_interfaces);

      // Test coupled interfaces
      // Check if the mesh was read correctly
      specfem::kokkos::HostView1d<int> h_elastic_ispec;
      specfem::kokkos::HostView1d<int> h_acoustic_ispec;
      specfem::kokkos::HostView1d<int> h_poroelastic_ispec;

      if (Test.databases.elastic_acoustic.elastic_ispec_file != "NULL") {
        h_elastic_ispec = coupled_interfaces.elastic_acoustic.h_elastic_ispec;
        specfem::testing::test_array(
            h_elastic_ispec, Test.databases.elastic_acoustic.elastic_ispec_file,
            coupled_interfaces.elastic_acoustic.num_interfaces);

        h_acoustic_ispec = coupled_interfaces.elastic_acoustic.h_acoustic_ispec;
        specfem::testing::test_array(
            h_acoustic_ispec,
            Test.databases.elastic_acoustic.acoustic_ispec_file,
            coupled_interfaces.elastic_acoustic.num_interfaces);

        // test if the edges match
        test_edges(compute.h_ibool, compute.coordinates.coord,
                   coupled_interfaces.elastic_acoustic.h_elastic_ispec,
                   coupled_interfaces.elastic_acoustic.h_acoustic_ispec,
                   coupled_interfaces.elastic_acoustic.h_elastic_edge,
                   coupled_interfaces.elastic_acoustic.h_acoustic_edge);
      } else {
        ASSERT_TRUE(coupled_interfaces.elastic_acoustic.num_interfaces == 0);
      }
      // Check if the mesh was read correctly
      if (Test.databases.elastic_poroelastic.elastic_ispec_file != "NULL") {
        h_elastic_ispec =
            coupled_interfaces.elastic_poroelastic.h_elastic_ispec;
        specfem::testing::test_array(
            h_elastic_ispec,
            Test.databases.elastic_poroelastic.elastic_ispec_file,
            coupled_interfaces.elastic_poroelastic.num_interfaces);

        h_poroelastic_ispec =
            coupled_interfaces.elastic_poroelastic.h_poroelastic_ispec;
        specfem::testing::test_array(
            h_poroelastic_ispec,
            Test.databases.elastic_poroelastic.poroelastic_ispec_file,
            coupled_interfaces.elastic_poroelastic.num_interfaces);

        // test if the edges match
        test_edges(compute.h_ibool, compute.coordinates.coord,
                   coupled_interfaces.elastic_poroelastic.h_elastic_ispec,
                   coupled_interfaces.elastic_poroelastic.h_poroelastic_ispec,
                   coupled_interfaces.elastic_poroelastic.h_elastic_edge,
                   coupled_interfaces.elastic_poroelastic.h_poroelastic_edge);
      } else {
        ASSERT_TRUE(coupled_interfaces.elastic_poroelastic.num_interfaces == 0);
      }

      // Check if the mesh was read correctly
      if (Test.databases.acoustic_poroelastic.acoustic_ispec_file != "NULL") {
        h_poroelastic_ispec =
            coupled_interfaces.acoustic_poroelastic.h_poroelastic_ispec;
        specfem::testing::test_array(
            h_poroelastic_ispec,
            Test.databases.acoustic_poroelastic.poroelastic_ispec_file,
            coupled_interfaces.acoustic_poroelastic.num_interfaces);

        h_acoustic_ispec =
            coupled_interfaces.acoustic_poroelastic.h_acoustic_ispec;
        specfem::testing::test_array(
            h_acoustic_ispec,
            Test.databases.acoustic_poroelastic.acoustic_ispec_file,
            coupled_interfaces.acoustic_poroelastic.num_interfaces);

        // test if the edges match
        test_edges(compute.h_ibool, compute.coordinates.coord,
                   coupled_interfaces.acoustic_poroelastic.h_poroelastic_ispec,
                   coupled_interfaces.acoustic_poroelastic.h_acoustic_ispec,
                   coupled_interfaces.acoustic_poroelastic.h_poroelastic_edge,
                   coupled_interfaces.acoustic_poroelastic.h_acoustic_edge);
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
