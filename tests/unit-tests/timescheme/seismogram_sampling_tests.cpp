#include "../SPECFEM_Environment.hpp"
#include "specfem/runtime_configuration/setup.hpp"
#include "specfem/timescheme/newmark.hpp"
#include <gtest/gtest.h>
#include <utility>

class SeismogramSamplingTest
    : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(SeismogramSamplingTest, AllocationMatchesRecordingSchedule) {
  const auto [nstep, sampling_interval] = GetParam();
  auto parameters = YAML::Load(R"(
parameters:
  header:
    title: Seismogram sampling
    description: Allocation must include the sample at step zero.
  sources: sources.yaml
  databases:
    mesh-database: database.bin
  receivers:
    stations: STATIONS
    angle: 0
    seismogram-type: [displacement]
    nstep_between_samples: 1
  simulation-setup:
    solver:
      time-marching:
        time-scheme:
          type: Newmark
          dt: 0.001
          nstep: 1
    simulation-mode:
      forward:
        writer:
          seismogram:
            format: ascii
            directory: .
)");
  parameters["parameters"]["receivers"]["nstep_between_samples"] =
      sampling_interval;
  parameters["parameters"]["simulation-setup"]["solver"]["time-marching"]
            ["time-scheme"]["nstep"] = nstep;
  const specfem::runtime_configuration::setup setup(parameters);

  // Only the recording schedule is exercised; no mesh or wavefield is needed.
  specfem::assembly::fields<specfem::element::dimension_tag::dim2> fields;
  specfem::time_scheme::newmark<decltype(fields),
                                specfem::simulation::type::forward>
      scheme(fields, nstep, sampling_interval, 0.001, 0.0);

  for (const auto [istep, dt] : scheme.iterate_forward()) {
    if (scheme.compute_seismogram(istep)) {
      EXPECT_LT(scheme.get_seismogram_step(), setup.get_max_seismogram_step())
          << "Recording step " << istep;
      EXPECT_LT(scheme.get_seismogram_step(), scheme.get_max_seismogram_step())
          << "Recording step " << istep;
      scheme.increment_seismogram_step();
    }
  }
  EXPECT_EQ(scheme.get_seismogram_step(), setup.get_max_seismogram_step());
  EXPECT_EQ(scheme.get_seismogram_step(), scheme.get_max_seismogram_step());
}

INSTANTIATE_TEST_SUITE_P(SamplingIntervals, SeismogramSamplingTest,
                         ::testing::Values(std::pair{ 1, 4 }, std::pair{ 4, 4 },
                                           std::pair{ 5, 4 }, std::pair{ 8, 4 },
                                           std::pair{ 9, 4 },
                                           std::pair{ 5, 1 }));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
