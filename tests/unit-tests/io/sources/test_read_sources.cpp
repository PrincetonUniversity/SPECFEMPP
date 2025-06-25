#include "../../Kokkos_Environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "io/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// Local constants since these would be set by the simulation.
int nsteps = 100;
type_real dt = 0.01;
int tshift = 0;            // for the single sources we are reading!
type_real user_t0 = -10.0; // user defined t0

// Internal t0 is being fixed using the halfduration of the source

specfem::wavefield::simulation_field wavefield_type =
    specfem::wavefield::simulation_field::forward;
using SourceVectorType =
    std::vector<std::shared_ptr<specfem::sources::source> >;

const static std::unordered_map<std::string, SourceVectorType> expected = {
  { "Single Moment Tensor",
    { std::make_shared<specfem::sources::moment_tensor>(
        2000.0, 3000.0, 1.0, 1.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(
            nsteps, dt, 1.0, 30.0, 1.0e10, false),
        wavefield_type) } },
  { "Single Force",
    { std::make_shared<specfem::sources::force>(
        2500.0, 2500.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0,
                                                            5.0, 1.0e10, false),
        wavefield_type) } },
  { "Single Cosserat Force",
    { std::make_shared<specfem::sources::cosserat_force>(
        2500.0, 2500.0, 0.0, 1.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0,
                                                            0.0, 1e10, false),
        wavefield_type) } }
};

TEST(IO_TESTS, read_sources) {
  /**
   *  This test checks whether a moment tensor source is read correctly
   */

  std::string test_data_file = "io/sources/test_config.yaml";

  YAML::Node test_config = YAML::LoadFile(test_data_file);
  YAML::Node sources_configs = test_config["Tests"];

  for (const auto &source_config : sources_configs) {

    std::string source_file = source_config["sources"].as<std::string>();
    std::string key = source_config["name"].as<std::string>();

    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test read " << key << ":\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;

    auto [sources, _t0] = specfem::io::read_sources(
        source_file, nsteps, user_t0, dt, specfem::simulation::type::forward);

    if (expected.find(key) != expected.end()) {

      auto expected_sources = expected.at(key);

      ASSERT_EQ(sources.size(), expected_sources.size());

      for (size_t i = 0; i < sources.size(); ++i) {

        auto source = sources[i];
        auto expected_source = expected_sources[i];

        // Print source id
        std::cout << "Act. Source type: " << typeid(source).name() << "\n";
        std::cout << "Exp. Source type: " << typeid(expected_source).name()
                  << "\n";

        // Don't forget about the user defined t0 at the top of this file!
        if (*source != *expected_source) {
          std::cout << "Source mismatch at index " << i << ":\n"
                    << "Expected:\n"
                    << expected_source->print()
                    << "\n"
                       "!=\n"
                    << "Actual:\n"
                    << source->print() << "\n";
          FAIL();
        }
      }

    } else {
      FAIL() << "Unknown source type: " << key;
    }

    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[FINISHED]\033[0m Test read" << key << ".\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
  }
}
