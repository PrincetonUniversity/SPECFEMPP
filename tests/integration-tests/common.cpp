#include "common.hpp"
#include "specfem/io.hpp"
#include "specfem/mpi.hpp"
#include "yaml-cpp/yaml.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ostream>
#include <stdexcept>

void specfem::integration_tests::PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

std::vector<specfem::integration_tests::Case>
specfem::integration_tests::load_cases(const std::vector<std::string> &sections,
                                       const int nproc) {
  const YAML::Node root = YAML::LoadFile("configurations/tests.yaml");
  std::vector<Case> cases;
  for (const auto &section : sections) {
    for (const auto &entry : root[section]) {
      if (entry.second.as<int>() != nproc)
        continue;
      Case c;
      c.name = entry.first.as<std::string>();
      c.directory = "configurations/dim3/" + section + "/" + c.name;
      c.nproc = nproc;
      cases.push_back(c);
    }
  }
  std::sort(cases.begin(), cases.end(),
            [](const Case &a, const Case &b) { return a.name < b.name; });
  return cases;
}

specfem::integration_tests::CaseConfig
specfem::integration_tests::CaseConfig::load(const Case &c) {
  const std::string file = c.directory + "/config.yaml";
  YAML::Node node;
  try {
    node = YAML::LoadFile(file);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load config file: " + file + " - " +
                             e.what());
  }
  CaseConfig config;
  config.nproc = node["config"]["nproc"].as<int>();
  config.tolerance = node["config"]["tolerance"].as<double>();
  config.specfem_config =
      c.directory + "/" + node["databases"]["specfem_config"].as<std::string>();
  config.traces =
      c.directory + "/" + node["databases"]["traces"].as<std::string>();
  return config;
}

std::vector<specfem::integration_tests::Station>
specfem::integration_tests::read_stations(const std::string &stations_file) {
  std::ifstream file(stations_file);
  if (!file)
    throw std::runtime_error("Cannot open stations file " + stations_file);
  std::vector<Station> stations;
  // A line is `STATION NETWORK latitude longitude elevation burial`
  std::string name, network, rest;
  while (file >> name >> network && std::getline(file, rest))
    stations.push_back({ network, name });
  return stations;
}

specfem::integration_tests::Trace
specfem::integration_tests::read_trace(const std::filesystem::path &file) {
  std::ifstream stream(file);
  if (!stream)
    throw std::runtime_error("Cannot open seismogram " + file.string());
  Trace trace;
  double time, value;
  while (stream >> time >> value) {
    trace.time.push_back(time);
    trace.value.push_back(value);
  }
  return trace;
}

specfem::integration_tests::RunDirectory::RunDirectory(const std::string &name)
    : path_(std::filesystem::absolute("integration_test_runs") / name) {
  if (specfem::MPI::main_proc()) {
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  specfem::MPI::sync_all();
}

void specfem::integration_tests::expect_matches_reference_traces(
    const std::filesystem::path &simulated, const std::string &reference,
    const std::vector<Station> &stations,
    const std::vector<specfem::enums::wavefield> &seismogram_types,
    const type_real dt, const double tolerance) {
  specfem::io::impl::ChannelGenerator channel_generator(dt);

  for (const auto &station : stations) {
    double error = 0.0;
    double reference_norm = 0.0;

    for (const auto seismogram_type : seismogram_types) {
      for (const auto &filename : channel_generator.get_station_filenames(
               station.network, station.name, "S3", seismogram_type)) {
        const Trace computed = read_trace(simulated / filename);
        const Trace expected =
            read_trace(std::filesystem::path(reference) / filename);

        ASSERT_LE(computed.value.size(), expected.value.size())
            << station.network << "." << station.name << " " << filename
            << ": fewer reference samples than simulated samples";

        for (std::size_t step = 0; step < computed.value.size(); ++step) {
          ASSERT_LE(std::abs(computed.time[step] - expected.time[step]), 1e-3)
              << filename << ": times do not match at sample " << step;
          const double difference = computed.value[step] - expected.value[step];
          error += difference * difference;
          reference_norm += expected.value[step] * expected.value[step];
        }
      }
    }

    error = std::sqrt(error);
    reference_norm = std::sqrt(reference_norm);

    // When the reference norm is zero both the reference and computed
    // seismograms must be zero for the test to pass.
    const bool failed = (reference_norm == 0)
                            ? (error > tolerance)
                            : (error / reference_norm > tolerance ||
                               std::isnan(error / reference_norm));
    EXPECT_FALSE(failed) << station.network << "." << station.name << ": error "
                         << error << " relative to norm " << reference_norm
                         << " exceeds tolerance " << tolerance;
  }
}
