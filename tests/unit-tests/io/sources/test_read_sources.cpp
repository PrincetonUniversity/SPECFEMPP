#include "../../Kokkos_Environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "io/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
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
using SourceVector2DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::dimension::type::dim2> > >;
using SourceVector3DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::dimension::type::dim3> > >;

const static std::unordered_map<std::string, SourceVector2DType> expected_2d = {
  { "2D Single Moment Tensor",
    { std::make_shared<
        specfem::sources::moment_tensor<specfem::dimension::type::dim2> >(
        2000.0, 3000.0, 1.0, 1.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(
            nsteps, dt, 1.0, 30.0, 1.0e10, false),
        wavefield_type) } },
  { "2D Single Force",
    { std::make_shared<
        specfem::sources::force<specfem::dimension::type::dim2> >(
        2500.0, 2500.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0,
                                                            5.0, 1.0e10, false),
        wavefield_type) } },
  { "2D Single Cosserat Force",
    { std::make_shared<
        specfem::sources::cosserat_force<specfem::dimension::type::dim2> >(
        2500.0, 2500.0, 0.0, 1.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0,
                                                            0.0, 1e10, false),
        wavefield_type) } }
};

const static std::unordered_map<std::string, SourceVector3DType> expected_3d = {
  { "3D Single Force",
    { std::make_shared<
        specfem::sources::force<specfem::dimension::type::dim3> >(
        2500.0, 2500.0, 2500.0, 0.0, 0.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(nsteps, dt, 10.0,
                                                            5.0, 1.0e10, false),
        wavefield_type) } },
  { "3D Single Moment Tensor",
    { std::make_shared<
        specfem::sources::moment_tensor<specfem::dimension::type::dim3> >(
        2000.0, 3000.0, 2000.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        std::make_unique<specfem::forcing_function::Ricker>(
            nsteps, dt, 1.0, 30.0, 1.0e10, false),
        wavefield_type) } }
};

TEST(IO_TESTS, read_2d_sources) {
  /**
   *  This test checks whether 2D sources are read correctly
   */

  std::string test_data_file = "io/sources/test_config.yaml";

  YAML::Node test_config = YAML::LoadFile(test_data_file);
  YAML::Node sources_configs = test_config["2D Tests"];

  for (const auto &source_config : sources_configs) {

    std::string source_file = source_config["sources"].as<std::string>();
    std::string key = source_config["name"].as<std::string>();

    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test read " << key << ":\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;

    auto [sources, _t0] = specfem::io::read_2d_sources(
        source_file, nsteps, user_t0, dt, specfem::simulation::type::forward);

    if (expected_2d.find(key) != expected_2d.end()) {

      auto expected_sources = expected_2d.at(key);

      ASSERT_EQ(sources.size(), expected_sources.size());

      for (size_t i = 0; i < sources.size(); ++i) {

        auto source = sources[i];
        auto expected_source = expected_sources[i];

        // Print source id
        std::cout << "Act. Source type: " << typeid(source).name() << "\n";
        std::cout << "Exp. Source type: " << typeid(expected_source).name()
                  << "\n";

        // Since we have comparison operators defined, we can use them
        EXPECT_EQ(*source, *expected_source)
            << "Source mismatch at index " << i << ":\n"
            << "Expected:\n"
            << expected_source->print()
            << "\n"
               "!=\n"
            << "Actual:\n"
            << source->print() << "\n";
      }
    } else {
      FAIL() << "Source not in expected list: " << key;
    }

    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[FINISHED]\033[0m Test read" << key << ".\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
  }
}

TEST(IO_TESTS, read_3d_sources) {
  /**
   *  This test checks whether 3D sources are read correctly
   */

  std::string test_data_file = "io/sources/test_config.yaml";

  YAML::Node test_config = YAML::LoadFile(test_data_file);
  YAML::Node sources_configs = test_config["3D Tests"];

  for (const auto &source_config : sources_configs) {

    std::string source_file = source_config["sources"].as<std::string>();
    std::string key = source_config["name"].as<std::string>();

    auto [sources, _t0] = specfem::io::read_3d_sources(
        source_file, nsteps, user_t0, dt, specfem::simulation::type::forward);

    if (expected_3d.find(key) != expected_3d.end()) {

      auto expected_sources = expected_3d.at(key);

      ASSERT_EQ(sources.size(), expected_sources.size());

      for (size_t i = 0; i < sources.size(); ++i) {

        SCOPED_TRACE("Testing source: " + key);

        auto source = sources[i];
        auto expected_source = expected_sources[i];

        // Print source id
        std::cout << "Act. Source type: " << typeid(source).name() << "\n";
        std::cout << "Exp. Source type: " << typeid(expected_source).name()
                  << "\n";

        // Since we have comparison operators defined, we can use them
        EXPECT_EQ(*source, *expected_source)
            << "Source mismatch at index " << i << ":\n"
            << "Expected:\n"
            << expected_source->print()
            << "\n"
               "!=\n"
            << "Actual:\n"
            << source->print() << "\n";
      }
    } else {
      FAIL() << "Source not in expected list: " << key;
    }
  }
}
