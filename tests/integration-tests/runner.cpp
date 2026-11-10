// Runs full 3D simulations through specfem::program::program_3d and compares
// the seismograms they write with reference traces.
//
// The cases are listed in configurations/tests.yaml as `<name>: <nproc>`, in
// sections that are directories of configurations/dim3. SPECFEM_TEST_SECTION
// and SPECFEM_TEST_NPROC select the cases of this executable: the section and
// the MPI size it is launched with.
//
// A case whose config.yaml has an `injection` entry also runs with the
// plane-wave plugin SPECFEM_INJECTION_TEST_PLUGIN; the comparison is named by
// its `compare` entry.

#include "SPECFEM_Environment.hpp"
#include "common.hpp"
#include "specfem/injection/options.hpp"
#include "specfem/io.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/dim3/program.hpp"
#include "specfem/runtime_configuration.hpp"
#include "yaml-cpp/yaml.h"
#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifndef SPECFEM_TEST_NPROC
#define SPECFEM_TEST_NPROC 1
#endif

#ifndef SPECFEM_TEST_SECTION
#define SPECFEM_TEST_SECTION "explicit_solver"
#endif

class Runner
    : public ::testing::TestWithParam<specfem::integration_tests::Case> {};

TEST_P(Runner, Run) {
  if (!SPECFEMEnvironment::IsMPISizeValid())
    GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();

  const auto &test_case = GetParam();
  const auto config = specfem::integration_tests::CaseConfig::load(test_case);
  EXPECT_EQ(config.nproc, test_case.nproc)
      << test_case.name << " is listed in tests.yaml with " << test_case.nproc
      << " processes but its config.yaml declares " << config.nproc;

  const specfem::integration_tests::RunDirectory directory(test_case.name);

  // Runs the case, with or without its injection, and returns the parameters
  // it ran with.
  const auto run = [&](const std::filesystem::path &output,
                       const bool with_injection) {
    YAML::Node parameters = YAML::LoadFile(config.specfem_config);
    parameters["parameters"]["simulation-setup"]["simulation-mode"]["forward"]
              ["writer"]["seismogram"]["directory"] = output.string();
    if (!with_injection)
      parameters["parameters"].remove("injection");

    if (specfem::MPI::main_proc())
      std::filesystem::create_directories(output);
    specfem::MPI::sync_all();

    std::optional<specfem::injection::PluginOptions> plugin;
#ifdef SPECFEM_INJECTION_TEST_PLUGIN
    if (with_injection && !config.plugin_config.empty())
      plugin = specfem::injection::PluginOptions{
        SPECFEM_INJECTION_TEST_PLUGIN,
        std::filesystem::absolute(config.plugin_config).string()
      };
#endif
    specfem::program::program_3d(parameters, {}, false, plugin);
    specfem::MPI::sync_all();
    return parameters;
  };

  const auto simulated = directory.path() / "simulated";
  const YAML::Node parameters = run(simulated, true);

  if (!config.plugin_config.empty()) {
    // Ranks whose part of the mesh has no injection face
    const auto database = specfem::MPI::format_proc_filename(
        parameters["parameters"]["databases"]["mesh-database"]
            .as<std::string>());
    const auto mesh =
        specfem::io::read_3d_mesh(database, specfem::attenuation::Setup{});
    int empty = mesh.boundaries.injection_surface.nfaces == 0 ? 1 : 0;
#ifdef SPECFEM_ENABLE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &empty, 1, MPI_INT, MPI_SUM,
                  specfem::MPI::communicator());
#endif
    if (specfem::MPI::main_proc())
      std::cout << test_case.name << ": " << empty << " of " << test_case.nproc
                << " ranks own no injection points\n";
  }

  std::optional<std::filesystem::path> without_injection;
  if (config.compare == "no-injection") {
    without_injection = directory.path() / "without_injection";
    run(*without_injection, false);
  }

  if (!specfem::MPI::main_proc())
    return;

  const specfem::runtime_configuration::setup setup(parameters);
  const auto stations = specfem::integration_tests::read_stations(
      parameters["parameters"]["receivers"]["stations"].as<std::string>());

  if (config.compare == "fortran") {
    specfem::integration_tests::expect_matches_reference_traces(
        simulated, config.traces, stations, setup.get_seismogram_types(),
        setup.get_dt(), config.tolerance);
  } else if (config.compare == "analytic" || config.compare == "serial") {
    specfem::integration_tests::expect_matches_reference_curves(
        simulated, config.traces, stations, setup.get_dt(), config.tolerance,
        test_case.name);
  } else if (config.compare == "no-injection") {
    specfem::integration_tests::expect_bitwise_equal(
        simulated, *without_injection, stations);
  } else {
    FAIL() << test_case.name << ": unknown comparison '" << config.compare
           << "' in its config.yaml";
  }
}

// Instance names are the case names: CMake registers one CTest entry per case.
INSTANTIATE_TEST_SUITE_P(
    Integration, Runner,
    ::testing::ValuesIn(specfem::integration_tests::load_cases(
        { SPECFEM_TEST_SECTION }, SPECFEM_TEST_NPROC)),
    [](const ::testing::TestParamInfo<specfem::integration_tests::Case> &info) {
      return info.param.name;
    });

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new SPECFEMEnvironment(SPECFEM_TEST_NPROC));
  return RUN_ALL_TESTS();
}
