#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "compute/assembly/assembly.hpp"
#include "mesh/mesh.hpp"
#include "parallel_configuration/range_config.hpp"
#include "policies/range.hpp"
#include "quadrature/quadratures.hpp"
#include "receiver/receiver.hpp"
#include "source/source.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {
struct configuration {
public:
  int processors;
};

void operator>>(YAML::Node &Node, configuration &configuration) {
  configuration.processors = Node["nproc"].as<int>();
  return;
}

struct databases {
public:
  databases() : processors(0){};
  databases(const int &nproc) : processors(nproc), filenames(nproc){};

  void append(const YAML::Node &Node) {
    filenames[Node["processor"].as<int>()] = Node["filename"].as<std::string>();
  }
  int processors;
  std::vector<std::string> filenames;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node config = Node["config"];
    config >> configuration;
    YAML::Node database = Node["databases"];

    assert(database.IsSequence());
    assert(database.size() == configuration.processors);

    databases = test_configuration::databases(configuration.processors);

    assert(databases.filenames.size() == configuration.processors);

    for (auto N : database)
      databases.append(N);

    assert(databases.filenames.size() == configuration.processors);

    return;
  }

  std::string name;
  std::string description;
  test_configuration::databases databases;
  test_configuration::configuration configuration;
};
} // namespace test_configuration

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_configuration::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_configuration::Test(N));

  return;
}

// ------------------------------------------------------------------------

template <typename ParallelConfig> auto execute_range_policy(const int nglob) {
  using PolicyType = specfem::policy::range<ParallelConfig>;
  PolicyType policy(nglob);
  using TestViewType = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  TestViewType test_view("test_view", nglob);
  TestViewType::HostMirror test_view_host =
      Kokkos::create_mirror_view(test_view);

  // initialize test_view
  Kokkos::parallel_for(
      "initialize_test_view",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nglob),
      KOKKOS_LAMBDA(const int iglob) { test_view(iglob) = 0; });

  Kokkos::fence();

  Kokkos::parallel_for(
      "execute_range_policy",
      static_cast<typename PolicyType::policy_type>(policy),
      KOKKOS_LAMBDA(const int iglob) {
        const auto iterator = policy.range_iterator(iglob);
        const auto index = iterator(0);

        constexpr bool using_simd = PolicyType::simd::using_simd;

        if constexpr (using_simd) {
          using mask_type = typename PolicyType::simd::mask_type;
          mask_type mask(
              [&](std::size_t lane) { return index.index.mask(lane); });
          using tag_type = typename PolicyType::simd::tag_type;
          using datatype = typename PolicyType::simd::datatype;
          datatype data;
          Kokkos::Experimental::where(mask, data)
              .copy_from(&test_view(index.index.iglob), tag_type());

          data += static_cast<type_real>(1);
          Kokkos::Experimental::where(mask, data)
              .copy_to(&test_view(index.index.iglob), tag_type());
        } else if constexpr (!using_simd) {
          test_view(index.index.iglob) += 1;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(test_view_host, test_view);
  return test_view_host;
}

TEST(POLICIES, RangePolicy) {
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
  std::string config_filename =
      "../../../tests/unit-tests/policies/test_config.yaml";
  std::vector<test_configuration::Test> Tests;
  parse_test_config(YAML::LoadFile(config_filename), Tests);

  std::vector<std::shared_ptr<specfem::sources::source> > sources;
  std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};

    return specfem::quadrature::quadratures(gll);
  }();
  std::vector<specfem::enums::seismogram::type> seismogram_types;

  for (auto &Test : Tests) {
    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
    specfem::mesh::mesh mesh(
        Test.databases.filenames[Test.configuration.processors - 1], mpi);
    specfem::compute::assembly assembly(mesh, quadrature, sources, receivers,
                                        seismogram_types, 0.0, 0.0, 0, 0,
                                        specfem::simulation::type::forward);

    const int nglob = compute_nglob(assembly.mesh.points.h_index_mapping);
    const int nspec = assembly.mesh.points.nspec;
    const int ngllz = assembly.mesh.points.ngllz;
    const int ngllx = assembly.mesh.points.ngllx;
    const auto index_mapping = assembly.mesh.points.h_index_mapping;

    using ParallelConfig = specfem::parallel_config::default_range_config<
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;
    using SimdParallelConfig = specfem::parallel_config::default_range_config<
        specfem::datatype::simd<type_real, true>,
        Kokkos::DefaultExecutionSpace>;

    const auto check_test_view = [&](const auto &test_view, std::string error) {
      for (int iglob = 0; iglob < nglob; iglob++) {
        if (test_view(iglob) != 1) {
          ADD_FAILURE();

          std::cout << "--------------------------------------------------\n"
                    << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
                    << "\n"
                    << "- Error: " << error << "\n"
                    << "  Index: \n "
                    << "    iglob = " << iglob << "\n"
                    << "--------------------------------------------------\n\n"
                    << std::endl;
          return;
        }
      }

      for (int ispec = 0; ispec < nspec; ispec++) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            const int iglob = index_mapping(ispec, iz, ix);
            if (test_view(iglob) != 1) {
              ADD_FAILURE();

              std::cout
                  << "--------------------------------------------------\n"
                  << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
                  << "\n"
                  << "- Error: " << error << "\n"
                  << "  Index: \n "
                  << "    ispec = " << ispec << "\n"
                  << "    iz = " << iz << "\n"
                  << "    ix = " << ix << "\n"
                  << "--------------------------------------------------\n\n"
                  << std::endl;
              return;
            }
          }
        }
      }
    };

    auto test_view = execute_range_policy<ParallelConfig>(nglob);
    auto simd_test_view = execute_range_policy<SimdParallelConfig>(nglob);

    check_test_view(test_view, "Error in RangePolicy with SIMD OFF");
    check_test_view(simd_test_view, "Error in RangePolicy with SIMD ON");

    std::cout << "--------------------------------------------------\n"
              << "\033[0;32m[PASSED]\033[0m Test name: " << Test.name << "\n"
              << "--------------------------------------------------\n\n"
              << std::endl;
  }

  return;
}

// ------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
