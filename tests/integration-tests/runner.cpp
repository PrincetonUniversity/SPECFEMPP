// Runs full 3D simulations through specfem::program::program_3d and compares
// the seismograms they write with reference traces.
//
// The cases are listed in configurations/tests.yaml as `<name>: <nproc>`.
// SPECFEM_TEST_NPROC is the MPI size of this executable; it runs only the cases
// that declare that size.

#include "SPECFEM_Environment.hpp"
#include "common.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/dim3/program.hpp"
#include "specfem/runtime_configuration.hpp"
#include "yaml-cpp/yaml.h"
#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifndef SPECFEM_TEST_NPROC
#define SPECFEM_TEST_NPROC 1
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

  const specfem::integration_tests::RunDirectory run(test_case.name);

  YAML::Node parameters = YAML::LoadFile(config.specfem_config);
  parameters["parameters"]["simulation-setup"]["simulation-mode"]["forward"]
            ["writer"]["seismogram"]["directory"] = run.path().string();

  specfem::program::program_3d(parameters, {});
  specfem::MPI::sync_all();

  if (!specfem::MPI::main_proc())
    return;

  const specfem::runtime_configuration::setup setup(parameters);
  specfem::integration_tests::expect_matches_reference_traces(
      run.path(), config.traces,
      specfem::integration_tests::read_stations(
          parameters["parameters"]["receivers"]["stations"].as<std::string>()),
      setup.get_seismogram_types(), setup.get_dt(), config.tolerance);
}

// Instance names are the case names: CMake registers one CTest entry per case.
INSTANTIATE_TEST_SUITE_P(
    Integration, Runner,
    ::testing::ValuesIn(specfem::integration_tests::load_cases(
        { "explicit_solver" }, SPECFEM_TEST_NPROC)),
    [](const ::testing::TestParamInfo<specfem::integration_tests::Case> &info) {
      return info.param.name;
    });

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new SPECFEMEnvironment(SPECFEM_TEST_NPROC));
  return RUN_ALL_TESTS();
}
